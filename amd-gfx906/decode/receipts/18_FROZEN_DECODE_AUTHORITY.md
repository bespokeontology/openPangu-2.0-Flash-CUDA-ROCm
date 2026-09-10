# 18 - FROZEN DECODE AUTHORITY (plain target decode, MTP OFF)

| item | value |
|---|---|
| binary | p92_gen (frozen copy: freeze-69.9/p92_gen) |
| sha256 | b28f0eef71121a0199315dfc4b5d1deee1ebbe3b8dafaad4cd40657a24d52d75 |
| result | 69.92 tok/s steady decode (28.90 -> 15.97 ms/token over the ladder, 1.81x) |
| decode mode | plain native target decode; the binary contains no MTP or speculative path |
| measurement | fixed token stream, context 512, 12 generated tokens |
| ladder rung | 69.92 mla_g16 branchless loads, 6 vmcnt(0) barriers -> 2 (ENGINEERING_REPORT.md optimization ladder) |
| supporting receipts | receipts/14_MLA_G16_ACCOUNTING.md, receipts/10_GFX906_PLAYBOOK.md |
| status | frozen authority of this release; supersedes the earlier 63.18 rung |

The 63.18 tok/s figure in ENGINEERING_REPORT.md section 1 is an EARLIER rung of the
same ladder and does not supersede this one. Every rung is a named physical mechanism;
the ladder is monotone in time, not a set of alternatives.
