> **DRAFT — not filed.** Proposed text for a GitHub issue against
> `EnzymeAD/Enzyme-JAX`. Written by the author of the accompanying patch on
> `ctessum-claude/Enzyme-JAX:perf/const-index-through-arith`; see `PR_DRAFT.md`.

# `slice_of_dynamic_update` / `dynamic_slice_to_static` never fire on Reactant-produced IR, because the start index is `subtract(c, c)` rather than `constant`

## Summary

`SliceOfDynamicUpdate` and `DynamicSliceToStatic` (both on by default —
`slice_of_dynamic_update<1>` and `dynamic_slice_to_static<16>` in
`enzymexlaGetTransformPassesList`) test for a compile-time start index with

```cpp
DenseIntElementsAttr startattr;
if (!matchPattern(update_start, m_Constant(&startattr)))
  return failure();          // or: legal = false / continue
```

That test is **syntactic**. A `dynamic_update_slice` start index is an
*operand*, and a producer is free to hand it over as a small arithmetic
expression over constants. Reactant does exactly that: it lowers a 1-based
Julia index by emitting a `subtract` against a literal one. In the StableHLO
module Reactant hands to XLA for one reverse-mode step of our model, **182 of
182 `stablehlo.dynamic_update_slice` ops have a start index defined by
`stablehlo.subtract`, and 0 by `stablehlo.constant`**:

```mlir
%c_31 = stablehlo.constant dense<85177> : tensor<i32>
%c_32 = stablehlo.constant dense<1>     : tensor<i32>
%77   = stablehlo.subtract %c_31, %c_32 : tensor<i32>
%78   = stablehlo.dynamic_update_slice %0, %76, %77
      : (tensor<247423xf64>, tensor<7056xf64>, tensor<i32>) -> tensor<247423xf64>
```

So the entire "read through a `dynamic_update_slice`" family is silently
coupled to a *separate* pattern, `sub_const_prop`, having folded the index
first. **That coupling — not the absence of the rewrite — is the bug.**

To be precise about what is and is not claimed: in a fully stock pipeline
`sub_const_prop` is enabled, folds these indices, and `slice_of_dynamic_update`
does then fire. The defect is that a *reader-reducing* rewrite is only
reachable through a switch that also controls a dozen *chain-reshaping*
rewrites, with no way to separate them and no diagnostic when they part company.
It bites in two ways:

1. **Any pipeline that excludes the folder loses the rewrite, invisibly.**
   `excluded_passes` works by base name. We exclude
   `dynamic_update_to_concat` because it rewrites our in-place DUS chain into
   whole-buffer concatenates, and *also* `sub_const_prop`, because starving the
   rest of the family of constant indices is the only lever that reaches the
   ones we cannot name individually. Measured on our forward step at
   production grid, the pair is worth **3.32x** (174.8 -> 52.7 ms median,
   bit-identical results). Restoring `sub_const_prop` alone, with
   `dynamic_update_to_concat` still excluded, we measured at **0.67x** — a 33%
   regression — so "just don't exclude the folder" is not available to us: some
   other still-enabled DUS-family pattern picks the constants up and costs more
   than `slice_of_dynamic_update` saves. The one rewrite we want is the one we
   cannot have.
2. **As a general fragility.** Any index arriving before the folder runs, or
   spelled as `add`/`multiply`/a widening `convert`, has the same effect. The
   predicate should be semantic, not syntactic — then per-pattern exclusion,
   the intended mechanism, is sufficient on its own.

XLA does fold the index — `%constant.97 = s32[] constant(85176)` appears in the
optimized HLO — but by then the rewrite that could have used it is gone.
`algebraic_simplifier.cc`'s `HandleSlice` has no slice-through-DUS
simplification. So the constant is available in exactly the place where the
rewrite is not, and the rewrite exists in exactly the place where the constant
is not.

## Why this matters: reverse-mode stencil adjoints

This is not a cosmetic missed peephole. It is the dominant cost of
reverse-mode differentiation of any stencil code that assembles a flat
"observed" buffer with in-place writes.

The forward program builds one flat buffer with ~90 in-place
`dynamic_update_slice`s. Enzyme's reverse of a DUS is:

* cotangent of the *update* = a slice of the incoming cotangent at the window;
* cotangent of the *operand* = the incoming cotangent **with that window
  zeroed**.

So the adjoint contains a chain of ~90 whole-buffer zeroing DUSes, each version
read back by many *static* 91-wide slices (the stencil cotangent width):

```
%dynamic-update-slice.0 = f64[247423]{0} dynamic-update-slice(
                              %param_1.335, %broadcast.1303, %constant.240)
%broadcast.1303          = f64[91]{0} broadcast(%constant.241)   # zeros
%constant.240            = s32[] constant(106454)                # literal offset
```

Because each intermediate version has many live readers, writing version k+1 in
place would clobber version k, and copy insertion duplicates the whole 1.98 MB
buffer once per live version. Measured, in one `rhs_vjp` module at our
production grid (13x7x72, CONUS):

| | |
|---|---|
| `stablehlo.dynamic_update_slice` on the 247423-element buffer | 180 (91 with a zero update = the reverse chain) |
| DUS start indices that are `stablehlo.constant` | **0** |
| DUS start indices that are `subtract(constant, constant)` | **182** |
| `dynamic-update-slice` instructions, optimized HLO | 2,319 |
| whole-buffer `f64[247423]` copies, optimized HLO | 98 |
| all buffer copies, optimized HLO | 252 |

And per call, from our own instrumented census of the same program:
`ssp_vjp` writes 217.36 M elements (1739 MB) against the primal `ssp_step`'s
10.43 M (83 MB), at the same ~5.6 GB/s — it is write-bandwidth-bound, not
arithmetic-bound — of which **394 whole-buffer copies = 779.9 MB = 45% of the
write traffic**. The transport adjoint costs 12–18x its primal as a result.

## The rewrite that is already here, and would fix it

`SliceOfDynamicUpdate` already implements precisely the needed rule:

```
slice(dynamic_update_slice(x, u, c), [a:b])
    -> slice(x, [a:b])            if [a,b) n [c, c+len(u)) = 0
    -> slice(u, [a-c : b-c])      if [a,b) subset of [c, c+len(u))
```

Applied transitively, every read is forwarded past the non-overlapping zeroing
DUSes to the base buffer. Classifying every reader of the 91 zeroing DUSes in
the real module against its zeroed window:

| reader | count |
|---|---|
| `stablehlo.slice`, **disjoint** from the zeroed window | **304** |
| `stablehlo.slice`, contained in it | 1 |
| `stablehlo.slice`, straddling its boundary | 6 |
| `stablehlo.dynamic_slice`, all with compile-time-known starts, all disjoint | **75** |
| the next `dynamic_update_slice` in the chain | 78 |
| other (`add`, `reshape`) | 6 |

So 380 of 386 non-chain readers are forwardable by the existing rule (the
`dynamic_slice`s via `dynamic_slice_to_static`, which is blocked by the *same*
predicate). Recomputing readers-per-version after forwarding:

| live readers per version | before | after |
|---|---|---|
| 0 | — | 2 |
| 1 | 8 | **88** |
| 2 | 76 | 1 |
| 3 | 1 | — |
| 4 | 3 | — |
| 74 | 2 | — |
| 147 | 1 | — |

**88 of 91 versions drop to exactly one consumer — the next DUS in the chain.**
The chain becomes a linear in-place sequence and copy insertion has nothing to
preserve. This is the crux: the win comes from reducing *readers per version*,
not from reducing the *number of versions*. (We tried a producer-side change
that reduced versions without reducing readers; it moved copies 99 -> 102 and
was a net regression.)

At fixed bandwidth, removing all 780 MB of copies takes `ssp_vjp` from 1739 MB
to ~959 MB, i.e. ~310 ms -> ~177 ms, about **1.73x on the transport VJP with no
forward-side cost**. That bound partially overlaps an emitter-side change we
already landed on our side (394 -> 331 copies, 306.6 -> 254.9 ms), so the two
factors are not multiplicative.

## Reproducer

Minimal, independent of our model — the only difference from
`test/lit_tests/sliceofdynamicupdateslice.mlir` is that the index is spelled as
a subtract:

```mlir
// enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=slice_of_dynamic_update" \
//   --transform-interpreter --enzyme-hlo-remove-transform
func.func @disjoint_sub(%operand: tensor<247423xf64>, %update: tensor<91xf64>)
    -> tensor<91xf64> {
  %one  = stablehlo.constant dense<1> : tensor<i32>
  %base = stablehlo.constant dense<106455> : tensor<i32>
  %off  = stablehlo.subtract %base, %one : tensor<i32>
  %dus  = stablehlo.dynamic_update_slice %operand, %update, %off
        : (tensor<247423xf64>, tensor<91xf64>, tensor<i32>) -> tensor<247423xf64>
  %s = stablehlo.slice %dus [99993:100084] : (tensor<247423xf64>) -> tensor<91xf64>
  return %s : tensor<91xf64>
}
```

Expected: `stablehlo.slice %operand [99993:100084]`. Actual, with only that
pattern in the set (current `main`, by inspection of the predicate): unchanged,
because it does not look through the `subtract`. In the full default pipeline
`sub_const_prop` folds the index first and the rewrite does happen — which is
the point: the rewrite is reachable only as a package deal with everything else
that a folded index enables.

## Proposed fix

Make the index predicate semantic. A small helper that evaluates a
single-element integer value through `stablehlo.constant`, single-element
`reshape`, widening integer-to-integer `convert`, and
`add`/`subtract`/`multiply` over compile-time-known operands — rejecting any
fold that overflows int64 or is not exactly representable in the op's own
element type — decouples these patterns from whichever const-folder happens to
be enabled. A patch doing that for the three predicates in
`SliceOfDynamicUpdate` and `DynamicSliceToStatic` is in `PR_DRAFT.md`.

The same syntactic test appears in `DUSSliceSimplify`, `SliceDUSToConcat`,
`DUSDUSSubsuming`, `DynamicUpdateToConcat`, `DynamicUpdateSliceElim`,
`DynamicUpdateSliceConstProp` and others; those can adopt the helper too, but
each of them rewrites the *shape* of the buffer chain rather than only reducing
readers, so they want their own benchmarking rather than a blanket change.

## Caveats from the reporter

* I cannot run the full downstream stack against a patched compiler. Reactant
  pins `Reactant_jll` as a prebuilt binary artifact, so validating end-to-end
  means a bazel XLA rebuild, which I did not attempt (no bazel or clang on this
  machine; the environment caps memory well below what such a build needs).
* Every IR statistic above was measured, by parsing the real dumped
  `module.mlir` and `*.cpu_after_optimizations.txt` from our production run —
  not estimated. The per-call byte and millisecond figures come from our own
  instrumentation of that same program.
* The ~1.73x is an arithmetic bound from bytes at fixed bandwidth, not an
  observed speedup. I have not been able to observe it.
