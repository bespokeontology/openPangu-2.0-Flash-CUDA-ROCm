// SINGLE SOURCE OF TRUTH for the MLA split geometry.
//
// P92_SPLITS was previously defined TWICE: unguarded as 16 in p92_mla_split.hip
// and behind an #ifndef as 8 in p92_mla_g16.hip. Production resolved to 16 only
// because p92_generate.hip includes every source into ONE translation unit and
// includes mla_split (line 23) before mla_g16 (line 24). Change either the
// include order or the single-TU build and the producer (k_p92_mla_g16) and the
// consumer (k_p92_mla_combine) would silently disagree on a value both use to
// index the same buffer.
//
// It was dormant, not benign. Proven consistent in the shipped binary on
// 2026-09-08: producer max pacc index 393215, consumer max 393215, capacity
// 393216 floats. See receipts/13_MLA_SPLIT_ABI.md.
#ifndef P92_MLA_GEOMETRY_H
#define P92_MLA_GEOMETRY_H

#define P92_SPLITS 16   // partial-softmax splits per head; grid is dim3(48, P92_SPLITS/4)

// Every consumer of this value derives from the macro; nothing may redefine it.
#ifdef P92_SPLITS_LOCKED
#error "P92_SPLITS must come only from p92_mla_geometry.h"
#endif
#define P92_SPLITS_LOCKED 1

#endif
