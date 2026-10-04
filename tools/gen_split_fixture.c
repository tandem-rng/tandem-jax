/* Writes tests/cross_port.json from the C reference. Build and run:
 *   cc -std=c99 -O2 -I../tandem-c tools/gen_split_fixture.c ../tandem-c/tandem.c -lm -o gen && ./gen > tests/cross_port.json */
#include <inttypes.h>
#include <stdio.h>
#include "tandem.h"

static void words(const tandem_rng *r) {
    uint32_t k[4];
    tandem_key(r, k);
    printf("[\"%08x\", \"%08x\", \"%08x\", \"%08x\"]", k[0], k[1], k[2], k[3]);
}

int main(void) {
    tandem_rng p = tandem_seed(42, 0, 32);
    const uint64_t idx[] = {0, 1, 2, 3, 1ull << 31, (1ull << 32) - 1, 1ull << 32, (1ull << 32) + 1, 1ull << 63, ~0ull};
    const uint64_t pos[] = {0, 127, 128, 1ull << 40};
    const uint64_t pur[] = {0, 7, 1ull << 63};
    printf("{\n  \"seed\": 42,\n  \"key\": ");
    words(&p);
    printf(",\n  \"split\": {");
    for (size_t i = 0; i < sizeof idx / sizeof *idx; i++) {
        tandem_rng c = tandem_split(&p, idx[i]);
        printf("%s\n    \"%" PRIu64 "\": ", i ? "," : "", idx[i]);
        words(&c);
    }
    printf("\n  },\n  \"fork\": {");
    for (size_t i = 0; i < sizeof pos / sizeof *pos; i++) {
        tandem_rng q = p, kids[3];
        tandem_set_position(&q, pos[i]);
        tandem_fork(&q, kids, 3);
        printf("%s\n    \"%" PRIu64 "\": {\"new_position\": %" PRIu64 ", \"children\": [", i ? "," : "", pos[i], tandem_position(&q));
        for (int j = 0; j < 3; j++) {
            printf(j ? ", " : "");
            words(&kids[j]);
        }
        printf("]}");
    }
    printf("\n  },\n  \"sub\": {");
    for (size_t i = 0; i < sizeof pur / sizeof *pur; i++) {
        tandem_rng c = tandem_sub(&p, pur[i]);
        printf("%s\n    \"%" PRIu64 "\": ", i ? "," : "", pur[i]);
        words(&c);
    }
    printf("\n  }\n}\n");
    return 0;
}
