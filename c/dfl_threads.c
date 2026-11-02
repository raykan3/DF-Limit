/* dfl_threads.c: a parallel loop on a pool of worker threads (pthreads or Windows threads), with a dynamic schedule
 * in chunks of two iterations, as in the Julia engine.  The number of workers defaults to the number of
 * processors; DFL_NTHREADS in the environment overrides it. */
#if !defined(_WIN32)
#define _GNU_SOURCE 1          /* clock_gettime and sysconf under -std=c11 (glibc) */
#define _DARWIN_C_SOURCE 1     /* (macOS) */
#endif
#include "dflimit.h"
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#endif

double dfl_wtime(void)
{
#ifdef _WIN32
    LARGE_INTEGER f, t; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t); return (double) t.QuadPart / (double) f.QuadPart;
#else
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec;
#endif
}

static int NWORKERS = 0;
static int default_workers(void)
{
    const char *e = getenv("DFL_NTHREADS");
    if (e && atoi(e) > 0) return atoi(e);
#ifdef _WIN32
    SYSTEM_INFO si; GetSystemInfo(&si); return (int) si.dwNumberOfProcessors;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN); return n > 0 ? (int) n : 1;
#endif
}
int dfl_nworkers(void) { if (NWORKERS == 0) NWORKERS = default_workers(); return NWORKERS; }
void dfl_set_nworkers(int n) { NWORKERS = n > 0 ? n : 1; }

typedef struct { int n; void (*body)(int, void *); void *ctx; volatile long next; } job_t;
#define CHUNK 2
#ifdef _WIN32
static long fetch_add(volatile long *p, long v) { return InterlockedExchangeAdd(p, v); }
static DWORD WINAPI worker(LPVOID arg)
#else
static long fetch_add(volatile long *p, long v) { return __sync_fetch_and_add(p, v); }
static void *worker(void *arg)
#endif
{
    job_t *jb = arg;
    for (;;) {
        long j0 = fetch_add(&jb->next, CHUNK);
        if (j0 >= jb->n) break;
        for (long j = j0; j < j0 + CHUNK && j < jb->n; j++) jb->body((int) j, jb->ctx);
    }
    return 0;
}
void dfl_parfor(int n, void (*body)(int j, void *ctx), void *ctx)
{
    int nw = dfl_nworkers();
    if (nw <= 1 || n <= 1) { for (int j = 0; j < n; j++) body(j, ctx); return; }
    if (nw > n) nw = n;
    job_t jb = {n, body, ctx, 0};
#ifdef _WIN32
    HANDLE *th = malloc((nw - 1) * sizeof(HANDLE));
    for (int w = 0; w < nw - 1; w++) th[w] = CreateThread(NULL, 0, worker, &jb, 0, NULL);
    worker(&jb);
    WaitForMultipleObjects(nw - 1, th, TRUE, INFINITE);
    for (int w = 0; w < nw - 1; w++) CloseHandle(th[w]);
#else
    pthread_t *th = malloc((nw - 1) * sizeof(pthread_t));
    for (int w = 0; w < nw - 1; w++) pthread_create(&th[w], NULL, worker, &jb);
    worker(&jb);
    for (int w = 0; w < nw - 1; w++) pthread_join(th[w], NULL);
#endif
    free(th);
}
