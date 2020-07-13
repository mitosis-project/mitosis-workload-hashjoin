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

#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <ctype.h>
#include <inttypes.h>
#include <limits.h>
#include <errno.h>

#include <string.h>
#include <fcntl.h>     /* open */
#include <unistd.h>    /* exit */
#include <sys/ioctl.h> /* ioctl */
#include <sys/mman.h>
#include <numa.h>
#include <time.h>
#include <pthread.h>
#include <signal.h>


#ifdef _OPENMP
#    include <omp.h>
#endif

#if (defined(GREEN_MARL) || defined(GUPS))
#    define DISABLE_SIGNAL_IN_MAIN
#endif

/* configuration options */

// use mfd instead of shm for creating the memory file
//#define CONFIG_USE_MFD 1

#define CONFIG_HAVE_MODIFIED_KERNEL 1

///< the name of the shared memory file created
#define CONFIG_SHM_FILE_NAME "/tmp/alloctest-bench"

///< virtual base address of the table to be mapped. Currently PML4[128 + x]
#define CONFIG_TABLE_MAP_BASE(x) (uintptr_t)((128UL + (x)) * (512UL << 30))

#define CONFIG_STATUS_PRINT stderr

#define CONFIG_MAX_ADDRESS_BITS 48

#define CONFIG_MMAP_FLAGS MAP_SHARED | MAP_FIXED | MAP_NORESERVE

#define CONFIG_MMAP_PROT PROT_READ | PROT_WRITE

#define CHECK_NODE(_n, _s)                                                                        \
    do {                                                                                          \
        if (_n < 0 || _n >= numa_num_configured_nodes()) {                                        \
            fprintf(stderr,                                                                       \
                    "WARNING: provided numa node for " _s " was "                                 \
                    " %d out of supported range! setting to 0.\n",                                \
                    _n);                                                                          \
            _n = 0;                                                                               \
        }                                                                                         \
    } while (0);

#ifndef MAP_HUGE_2MB
#    define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif


FILE *opt_file_out = NULL;  ///< standard output
static bool opt_interference = false;

#ifdef CONFIG_USE_MFD
#    include <linux/memfd.h>
#    include <asm/unistd_64.h>

static inline int memfd_create(const char *name, unsigned int flags)
{
    return syscall(__NR_memfd_create, name, flags);
}
#endif


/*
 * ============================================================================
 * PT Dump
 * ============================================================================
 */

static bool opt_dump_pt_crc = false;
static bool opt_dump_pt_full = false;
static void *opt_dump_buf = NULL;

int opt_numa_node_pte = 0;
int opt_madvise_advice = MADV_NOHUGEPAGE;
struct bitmask *opt_runmask = NULL;
struct bitmask *opt_allocmask = NULL;
struct bitmask *opt_ptemask = NULL;

extern int real_main(int argc, char *argv[]);


void signalhandler(int sig)
{
    fprintf(opt_file_out, "<sig>Signal %i caught!</sig>\n", sig);

    FILE *fd3 = fopen(CONFIG_SHM_FILE_NAME ".done", "w");

    if (fd3 == NULL) {
        fprintf(stderr, "ERROR: could not create the shared memory file descriptor\n");
        exit(-1);
    }

    usleep(250);

    fprintf(opt_file_out, "</benchmark>\n");

    exit(0);
}

#include <sys/time.h>

int main(int argc, char *argv[])
{
    struct timeval tstart, tend;
    gettimeofday(&tstart, NULL);

    for (int i = 0; i < argc; i++) {
        printf("%s ", argv[i]);
    }
    printf("\n");

    opt_file_out = stdout;

    int c;
    while ((c = getopt(argc, argv, "o:h")) != -1) {
        switch (c) {
        case '-':
            break;
        case 'h':
            printf("usage: %s [-p N] [-d N] [-r N] [-m N] [-o FILE]\n", argv[0]);
            printf("p: pagetable node, d: datanode, r: runnode, m: memory in GB\n");
            return 0;
        case 'o':
            opt_file_out = fopen(optarg, "a");
            if (opt_file_out == NULL) {
                fprintf(stderr, "Could not open the file '%s' switching to stdout\n", optarg);
                opt_file_out = stdout;
            }
            break;
        case '?':
            switch (optopt) {
            case 'o':
                fprintf(stderr, "Option -%c requires an argument.\n", optopt);
                return -1;
            default:
                fprintf(stderr, "Unknown option `-%c'.\n", optopt);
                return -1;
            }
        }
    }

    int prog_argc = 0;
    char **prog_argv = NULL;

    prog_argv = &argv[0];
    prog_argc = argc;

    optind = 1;

    for (int i = 0; i < argc; i++) {
        if (strcmp("--", argv[i]) == 0) {
            argv[i] = argv[0];
            prog_argv = &argv[i];
            prog_argc = argc - i;
            break;
        }
    }

    /* start with output */
    fprintf(opt_file_out, "<benchmark exec=\"%s\">\n", argv[0]);

    fprintf(opt_file_out, "<config>\n");
#ifdef _OPENMP
    fprintf(opt_file_out, "  <openmp>on</openmp>");
#else
    fprintf(opt_file_out, "  <openmp>off</openmp>");
#endif
    fprintf(opt_file_out, "</config>\n");

    struct sigaction sigact;
    sigset_t block_set;

    sigfillset(&block_set);
    sigdelset(&block_set, SIGUSR1);

    sigemptyset(&sigact.sa_mask);
    sigact.sa_flags = 0;
    sigact.sa_handler = signalhandler;
    sigaction(SIGUSR1, &sigact, NULL);

    fprintf(opt_file_out, "<run>\n");
    real_main(prog_argc, prog_argv);
    fprintf(opt_file_out, "</run>\n");

    FILE *fd3 = fopen(CONFIG_SHM_FILE_NAME ".done", "w");

    if (fd3 == NULL) {
        fprintf(stderr, "ERROR: could not create the shared memory file descriptor\n");
        exit(-1);
    }

    usleep(250);

    gettimeofday(&tend, NULL);
    printf("Took: %zu.%03zu\n", tend.tv_sec - tstart.tv_sec,
           (tend.tv_usec - tstart.tv_usec) / 1000);
    fprintf(opt_file_out, "Took: %zu.%03zu\n", tend.tv_sec - tstart.tv_sec,
            (tend.tv_usec - tstart.tv_usec) / 1000);

    fprintf(opt_file_out, "</benchmark>\n");
    fflush(opt_file_out);
    return 0;
}
