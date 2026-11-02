/* dfl_lock.h: a mutex for the caches, on pthreads or Windows */
#ifndef DFL_LOCK_H
#define DFL_LOCK_H
#ifdef _WIN32
#include <windows.h>
typedef SRWLOCK dfl_lock_t;
#define DFL_LOCK_INIT SRWLOCK_INIT
static inline void dfl_lock(dfl_lock_t *l) { AcquireSRWLockExclusive(l); }
static inline void dfl_unlock(dfl_lock_t *l) { ReleaseSRWLockExclusive(l); }
#else
#include <pthread.h>
typedef pthread_mutex_t dfl_lock_t;
#define DFL_LOCK_INIT PTHREAD_MUTEX_INITIALIZER
static inline void dfl_lock(dfl_lock_t *l) { pthread_mutex_lock(l); }
static inline void dfl_unlock(dfl_lock_t *l) { pthread_mutex_unlock(l); }
#endif
#endif
