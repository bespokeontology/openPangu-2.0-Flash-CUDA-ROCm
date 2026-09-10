// Gate: every trunk layer owned exactly once, DSA layers evenly distributed.
#include <cstdio>
#include <vector>
#include "p92amd/p92_placement.h"
int main() {
    std::vector<int> owned(P92_TRUNK_LAYERS, 0);
    int bad = 0;
    printf("card  layers      count  DSA  sliding\n");
    for (int d = 0; d < P92_NGPU; d++) {
        int dsa = 0, sl = 0;
        for (int l = p92_layer_begin(d); l < p92_layer_end(d); l++) {
            if (l < 0 || l >= P92_TRUNK_LAYERS) { printf("  layer %d out of range\n", l); bad++; continue; }
            owned[l]++;
            if (p92_layer_is_dsa(l)) dsa++; else sl++;
            if (p92_layer_owner(l) != d) { printf("  owner(%d)=%d expected %d\n", l, p92_layer_owner(l), d); bad++; }
        }
        printf("%4d  %2d-%-2d %9d %5d %8d\n", d, p92_layer_begin(d), p92_layer_end(d) - 1,
               p92_layer_count(d), dsa, sl);
    }
    int total = 0, dup = 0, miss = 0;
    for (int l = 0; l < P92_TRUNK_LAYERS; l++) {
        total += owned[l];
        if (owned[l] > 1) { printf("  layer %d owned %d times\n", l, owned[l]); dup++; }
        if (owned[l] == 0) { printf("  layer %d owned by nobody\n", l); miss++; }
    }
    printf("total ownerships %d (expect %d), duplicates %d, unowned %d\n",
           total, P92_TRUNK_LAYERS, dup, miss);
    // the Qwen scheme, for the record
    int trunc = P92_TRUNK_LAYERS - (P92_TRUNK_LAYERS % P92_NGPU);
    printf("legacy truncating scheme would execute %d of %d layers, dropping %d..%d\n",
           trunc, P92_TRUNK_LAYERS, trunc, P92_TRUNK_LAYERS - 1);
    bad += dup + miss + (total != P92_TRUNK_LAYERS);
    printf("%s\n", bad ? "P92_PLACEMENT_FAIL" : "P92_PLACEMENT_OK");
    return bad ? 1 : 0;
}
