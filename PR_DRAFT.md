> **DRAFT — no PR opened.** Proposed description for a pull request against
> `EnzymeAD/Enzyme-JAX` from
> `ctessum-claude/Enzyme-JAX:perf/const-index-through-arith`.
> Companion bug report: `ISSUE_DRAFT.md`.

# Match `dynamic_update_slice` / `dynamic_slice` start indices semantically

## What this changes

`SliceOfDynamicUpdate` and `DynamicSliceToStatic` need the start index of the
op they look through to be compile-time known, and test for it with
`matchPattern(idx, m_Constant(&attr))`. That test succeeds only on a literal
`stablehlo.constant`. Because a start index is an *operand*, producers
routinely spell it as arithmetic over constants — Reactant lowers a 1-based
Julia index as `stablehlo.subtract %c_start, %c_one`, so in the modules it
emits **every** index is a `subtract` and none is a `constant`.

In a stock pipeline `sub_const_prop` folds those indices first and the
patterns do fire; the problem is the coupling. A rewrite that only *reduces
readers* of a DUS is reachable only through a switch that simultaneously arms a
dozen rewrites that *reshape the buffer chain* — and on our workload the second
group costs more than the first saves, so we cannot enable the folder (see
"Deliberately not done" for the measurements). Making the predicate semantic
separates the two, so per-pattern exclusion, the intended mechanism, becomes
sufficient on its own.

This PR adds `mlir::enzyme::matchConstantIntScalar(Value, int64_t &, unsigned
maxDepth = 8)` and uses it for the three index predicates in those two
patterns. It matches a single-element integer value through:

* `stablehlo.constant` (and anything else `m_Constant` sees through);
* single-element `stablehlo.reshape`;
* `stablehlo.convert`, integer to integer, same signedness, widening only;
* `stablehlo.add` / `subtract` / `multiply` over operands that are themselves
  compile-time known.

No IR is created — this is a predicate, not a folder. Every fold is rejected
if it overflows `int64_t` *or* if the result is not exactly representable in
the op's own element type and signedness, so wraparound is never assumed;
`maxDepth` bounds recursion.

Diff: 3 files, ~93 insertions / 9 deletions, plus two lit tests.

## Correctness condition

The rewrite itself is unchanged and already in tree:

```
slice(dynamic_update_slice(x, u, c), [a:b])
    -> slice(x, [a:b])            if [a,b) n [c, c+len(u)) = 0   (disjoint)
    -> slice(u, [a-c : b-c])      if [a,b) subset of [c, c+len(u))  (contained)
```

It needs only that `c` is compile-time known and that the windows are
comparable in every dimension — which is what the existing code already checks
per dimension. Widening the predicate does not widen the rewrite: it only
widens the set of `c` the existing checks can see. `DynamicSliceToStatic` is
likewise unchanged in effect — it still requires `0 <= c` and
`c + size <= dim`.

## Why the copies disappear rather than move

The motivating case is the reverse mode of a stencil program that assembles a
flat observed buffer with ~90 in-place `dynamic_update_slice`s. Enzyme's
adjoint of a DUS gives the operand's cotangent as "the incoming cotangent with
the window zeroed", so the adjoint is a chain of ~90 whole-buffer zeroing
DUSes, and each intermediate version is read by many static slices. Every
version therefore has many live readers, writing version k+1 in place would
clobber version k, and copy insertion duplicates the entire 1.98 MB buffer once
per live version.

Forwarding the reads past non-overlapping windows removes the *readers*, and
that is what makes the copies unnecessary rather than relocated. Measured on
the real dumped module (13x7x72 CONUS, 91 zeroing DUSes on a 247423-element
buffer), live readers per version:

| live readers | before | after forwarding |
|---|---|---|
| 0 | — | 2 |
| 1 | 8 | **88** |
| 2 | 76 | 1 |
| 3 | 1 | — |
| 4 | 3 | — |
| 74 | 2 | — |
| 147 | 1 | — |

88 of 91 versions end with exactly one consumer — the next DUS in the chain —
so the chain is a linear in-place sequence with nothing for copy insertion to
preserve. Of the 386 non-chain readers, 304 static slices are disjoint from
the zeroed window, 1 is contained, 6 straddle, and 75 are `dynamic_slice`s
whose starts are all compile-time known and all disjoint (those need
`dynamic_slice_to_static`, blocked by the same predicate — hence both patterns
in one PR).

This distinguishes the change from a producer-side attempt we made and
rejected, which reduced the *number of versions* without reducing *readers per
version*: it moved copies 99 -> 102 and was a net regression. Readers are the
thing to reduce.

## Payoff, and its honest bound

In that module the reverse program writes 217.36 M elements (1739 MB) per call
against the primal's 10.43 M (83 MB) at the same ~5.6 GB/s, so it is
write-bandwidth-bound; 394 whole-buffer copies account for 779.9 MB = 45% of
the write traffic. At fixed bandwidth, removing them takes the call from
1739 MB to ~959 MB, i.e. ~310 ms -> ~177 ms, about **1.73x on the transport
VJP with no forward-side cost**.

That is an arithmetic bound from bytes, not an observed speedup, and it
partially overlaps a producer-side change we already landed (394 -> 331 copies,
306.6 -> 254.9 ms), so the two are not multiplicative.

## Tested vs. written

Plainly: **I wrote both lit tests and ran neither.** There is no bazel and no
clang on the machine this was developed on, and building
`enzymexlamlir-opt` means a bazel build of XLA — hours to days and tens of GB,
which I did not attempt. I looked for a cheaper harness (a prebuilt opt binary
in the Julia artifact tree, a system MLIR) and there is none.

What I did do:

* **Read** the current `SliceOfDynamicUpdate`, `DynamicSliceToStatic`,
  `SliceDUSToConcat`, `DUSSliceSimplify` and `binaryConstProp`, and the
  default pattern list in
  `src/enzyme_ad/jax/Integrations/c/EnzymeXLA.cpp`, to establish that the
  rewrite already exists, is enabled by default, and is blocked only by the
  predicate.
* **Measured** the claim on real IR: parsed the 67 MB `module.mlir` handed to
  PJRT and the 9.5 MB optimized-HLO dump from a production run and produced
  every count in the tables above (182/182 indices are `subtract`; 0 are
  `constant`; the reader classification; the before/after reader histogram).
* **Re-implemented `matchConstantIntScalar`'s exact fold set in Python** and
  ran it over that module, which is how the "all 91 indices resolvable" and
  "88 of 91 drop to one reader" numbers were obtained. That validates the
  *semantics* of the predicate against real input; it does not validate the
  C++.
* Wrote two lit tests in the project's conventions, each pinned to the single
  pattern under test so they cannot pass by accident via a const-folder:
  * `test/lit_tests/sliceofdynamicupdate_arithindex.mlir` — disjoint via
    `subtract`; contained via `subtract`; disjoint via `add` + widening
    `convert`; a three-deep zeroing chain forwarded to the base buffer; and a
    genuinely runtime index left alone.
  * `test/lit_tests/dynamicslicetostatic_arithindex.mlir` — `subtract` index
    becomes a static slice; runtime index left alone.

The C++ has not been compiled. Reviewers should assume the tests need their
`CHECK` lines adjusted on first run.

## Deliberately not done

The same syntactic predicate appears in `DUSSliceSimplify`,
`SliceDUSToConcat`, `DUSDUSSubsuming`, `DynamicUpdateToConcat`,
`DynamicUpdateSliceElim`, `DynamicUpdateSliceConstProp` and others, and they
can all adopt the helper. I left them alone on purpose. Those rewrites change
the *shape* of the buffer chain (DUS chain -> concatenates, pads, fused
windows) rather than only reducing readers, and at least one of them —
`dynamic_update_to_concat` — is a large pessimization on this workload: it
rewrites an in-place DUS chain into whole-buffer concatenates, costing us
**3.32x** on the forward step at production grid (174.8 -> 52.7 ms when
disabled, bit-identical results).

That is the related finding worth flagging. Because `excluded_passes` is by
base name and that whole family keys on a folded index, our workaround has been
to exclude `sub_const_prop` as well — starving the family of constants
wholesale. The pair is worth 3.32x to us. Restoring `sub_const_prop` alone,
with `dynamic_update_to_concat` still excluded, we measured at **0.67x**: some
other still-enabled DUS-family pattern picks the constants up and costs more
than `slice_of_dynamic_update` saves. So the coupling is not merely
inelegant — it makes the one rewrite we want unreachable. That is the whole
motivation for this PR, and the reason it widens exactly two predicates rather
than the family.

Reviewers should still weigh the converse risk: on a pipeline that had been
excluding `sub_const_prop`, these two patterns will now see compile-time
indices for the first time, and any workload that was (perhaps unknowingly)
relying on the reads *not* being forwarded will change shape. I believe that is
the correct default — forwarding a read past a provably disjoint write is
strictly less work — but it is a behaviour change for such pipelines, not a
no-op.

## Alternative considered: change Enzyme's reverse rule instead

The chain could be killed at its source: on differentiating a run of DUSes with
disjoint compile-time windows, emit **one** masked zeroing over the union
instead of ~90 whole-buffer versions.

I think the simplifier route in this PR is the better contribution, for four
reasons.

1. **Layer.** Enzyme's per-op reverse for DUS is already correct and minimal.
   The blowup is a liveness/copy-insertion artifact of *many readers per
   version*, not of the AD rule. Teaching an AD rule to pattern-match a chain
   shape puts a buffer-layout optimization in the differentiation layer.
2. **Generality.** The predicate fix repairs every "read through a DUS"
   simplification for every producer, forward and reverse. The AD fix repairs
   one adjoint shape, and does nothing for the *forward* chain, whose heads
   still carry 111 and 217 readers in our module (all `gather`s — a separate
   opportunity, since nothing currently forwards a `gather` past a DUS).
3. **Cost.** This PR is ~90 lines and no new IR. The union-mask form needs the
   union materialized — either a full-size constant mask for a `select`, or a
   run of DUSes that then has to be proven single-consumer, which is the same
   liveness problem again.
4. **It does not actually avoid the predicate problem.** Recognizing "disjoint
   constant windows" in the AD rule requires reading those same start indices,
   so it needs the same semantic matcher anyway.

The narrow AD-side change I would still support as a complement: when
reverse-differentiating a DUS whose operand value has a single use, zero in
place rather than producing a new version. That is a local, layer-appropriate
rule, and it composes with this PR rather than substituting for it.

## Also worth an upstream fix, elsewhere

`openxla/xla`'s `HandleSlice` in
`xla/hlo/transforms/simplifiers/algebraic_simplifier.cc` has no
slice-through-`dynamic-update-slice` simplification. In the post-XLA HLO the
index *is* a literal (`%constant.97 = s32[] constant(85176)` — XLA folds the
subtract), so the constant is available exactly where the rewrite is missing,
and the rewrite exists exactly where the constant is missing. Adding it there
would catch producers that bypass EnzymeXLA entirely. I have not written that
patch; fixing it here reaches consumers through a Reactant version bump instead
of an XLA release, which is why this is the PR.
