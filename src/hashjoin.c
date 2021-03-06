/**
 * MIT License
 * Copyright (c) 2020 Mitosis-Project
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include "config.h"
#include "murmur3.h"

#ifdef _OPENMP
#    include <omp.h>
#endif


///< this is a table element it has a key and a payload
struct element
{
    uint64_t key;
    uint64_t payload[CONFIG_ELEMENT_TUPLE_SIZE];
};

///< this is a hashtable element
struct htelm
{
    uint64_t key;
    struct htelm *next;
};


///< this function allocates mmeory with an alignment constraint
static void allocate(void **memptr, size_t size, size_t align)
{
    printf("allocating %zu MB memory with alignment %zu \n", size >> 20, align);
    if (posix_memalign(memptr, align, size)) {
        printf("ENOMEM\n");
        exit(1);
    }
    memset(*memptr, 0, size);
}


static inline uint64_t hash(uint64_t key)
{
    uint64_t out[2];
    MurmurHash3_x64_128(&key, sizeof(uint64_t), CONFIG_RAND_SEED, out);
    return out[0];
}

int real_main(int argc, char *argv[]);
int real_main(int argc, char *argv[])
{
    size_t hashsize = CONFIG_DEFAULT_HASH_SIZE;
    size_t nlookups = CONFIG_DEFAULT_NUM_LOOKUPS;
    size_t outersize = CONFIG_DEFAULT_OUTER_SIZE;
    size_t innersize = CONFIG_DEFAULT_INNER_SIZE;

    int c;
    while ((c = getopt(argc, argv, "s:n:o:i:")) != -1) {
        switch (c) {
        case 's':
            hashsize = strtol(optarg, NULL, 10);
            break;
        case 'n':
            nlookups = strtol(optarg, NULL, 10);
            break;
        case 'o':
            outersize = strtol(optarg, NULL, 10);
            break;
        case 'i':
            innersize = strtol(optarg, NULL, 10);
            ;
            break;
        default:
            printf("unknown option '%c'\n", c);
            return -1;
        }
    }

    printf("Hashtable Size: %zuMB\n", (hashsize * sizeof(struct htelm *)) >> 20);
    printf("Datatable Size Size: %zuMB\n", (outersize * sizeof(struct element)) >> 20);
    printf("Element Size: %zu MB\n", (hashsize * sizeof(struct htelm)) >> 20);
    printf("Total: %zu MB\n", ((hashsize * sizeof(struct htelm *))
                               + ((outersize + innersize) * sizeof(struct element))
                               + (hashsize * sizeof(struct htelm)))
                                  >> 20);

    /* allocate the hash table */
    struct htelm **hashtable;
    allocate((void **)&hashtable, hashsize * sizeof(struct htelm *),
             CONFIG_CACHELINE_SIZE);


    /* allocate the hash table elements for the join */
    struct htelm *htelms;
    allocate((void **)&htelms, innersize * sizeof(struct htelm), CONFIG_LARGE_PAGE_SIZE);


    size_t nconflicts = 0;
    for (size_t i = 0; i < innersize; i++) {
        htelms[i].key = (i + 1) * CONFIG_INNER_KEY_STRIDE;
        size_t idx = hash(htelms[i].key) % hashsize;

        nconflicts += (hashtable[idx] ? 1 : 0);

        /* insert it into the hash table */
        htelms[i].next = hashtable[idx];
        hashtable[idx] = &htelms[i];
    }

    /* build the outer table */
    struct element *table;
    allocate((void **)&table, outersize * sizeof(struct element), CONFIG_LARGE_PAGE_SIZE);
    for (size_t i = 0; i < outersize; i++) {
        table[i].key = i;
    }

    fprintf(stderr, "signalling readyness to %s\n", CONFIG_SHM_FILE_NAME ".ready");
    FILE *fd2 = fopen(CONFIG_SHM_FILE_NAME ".ready", "w");

    if (fd2 == NULL) {
        fprintf(stderr, "ERROR: could not create the shared memory file descriptor\n");
        exit(-1);
    }

    usleep(100);

    size_t matches = 0;
    size_t loaded = 0;

    struct timeval tstart, tend;
    gettimeofday(&tstart, NULL);

#ifdef _OPENMP
#    pragma omp parallel for reduction(+ : matches)
#endif
    for (size_t j = 0; j < nlookups; j++) {
        for (size_t i = 0; i < outersize; i++) {
            uint64_t hash[2]; /* Output for the hash */

            size_t idx = i;
            MurmurHash3_x64_128(&table[idx].key, sizeof(table[idx].key), CONFIG_RAND_SEED, hash);

            struct htelm *e = hashtable[hash[0] % hashsize];
            while (e) {
                if (e->key == table[idx].key) {
                    matches++;
                    break;
                }
                e = e->next;
            }

            e = hashtable[hash[1] % hashsize];
            while (e) {
                if (e->key == table[idx].key) {
                    matches++;
                    break;
                }
                e = e->next;
            }
        }
    }

    gettimeofday(&tend, NULL);
    fprintf(stderr, "signalling done to %s\n", CONFIG_SHM_FILE_NAME ".done");
    FILE *fd1 = fopen(CONFIG_SHM_FILE_NAME ".done", "w");

    if (fd1 == NULL) {
        fprintf(stderr, "ERROR: could not create the shared memory file descriptor\n");
        exit(-1);
    }

    printf("got %zu matches / %zu matches per iteration of %zu \n", matches, matches / nlookups, 2 * outersize);
    printf("hashtable conflicts = %zu\n", nconflicts);

    printf("Lookup time: %zu.%03zu\n", tend.tv_sec - tstart.tv_sec,
           (tend.tv_usec - tstart.tv_usec) / 1000);

    return 0;
}
