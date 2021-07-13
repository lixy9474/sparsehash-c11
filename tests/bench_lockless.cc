// Copyright (c) 2005, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// ---
// Authors: Sanjay Ghemawat and Craig Silverstein

// Time various hash map implementations
//
// Below, times are per-call.  "Memory use" is "bytes in use by
// application" as reported by tcmalloc, compared before and after the
// function call.  This does not really report fragmentation, which is
// not bad for the sparse* routines but bad for the dense* ones.
//
// The tests generally yield best-case performance because the
// code uses sequential keys; on the other hand, "map_fetch_random" does
// lookups in a pseudorandom order.  Also, "stresshashfunction" is
// a stress test of sorts.  It uses keys from an arithmetic sequence, which,
// if combined with a quick-and-dirty hash function, will yield worse
// performance than the otherwise similar "map_predict/grow."
//
// Consider doing the following to get good numbers:
//
// 1. Run the tests on a machine with no X service. Make sure no other
//    processes are running.
// 2. Minimize compiled-code differences. Compare results from the same
//    binary, if possible, instead of comparing results from two different
//    binaries.
//
// See PERFORMANCE for the output of one example run.

#include <cstdint>  // for uintptr_t
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

extern "C" {
#include <time.h>
#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif
#ifdef HAVE_SYS_RESOURCE_H
#include <sys/resource.h>
#endif
#ifdef HAVE_SYS_UTSNAME_H
#include <sys/utsname.h>
#endif  // for uname()
}

// The functions that we call on each map, that differ for different types.
// By default each is a noop, but we redefine them for types that need them.

#include <map>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <set>
#include <chrono>
#include <type_traits>
#include <sparsehash/dense_hash_map>
#include <sparsehash/sparse_hash_map>
#include <sparsehash/dense_hash_map_lockless>
#include "rwlock.h"

using std::map;
using std::unordered_map;
using std::swap;
using std::vector;
using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::time_point;
using std::chrono::nanoseconds;
using google::dense_hash_map;
using google::sparse_hash_map;
using google::dense_hash_map_lockless;
using std::hash;  // or __gnu_cxx::hash, or maybe tr1::hash, depending on your OS


static bool FLAGS_test_sparse_hash_map = true;
static bool FLAGS_test_dense_hash_map = true;
static bool FLAGS_test_hash_map = true;
static bool FLAGS_test_map = true;

static bool FLAGS_test_4_bytes = true;
static bool FLAGS_test_8_bytes = true;
static bool FLAGS_test_16_bytes = true;
static bool FLAGS_test_256_bytes = true;
static bool FLAGS_test_parallel = true;

static const int kDefaultIters = 10000000;
easy_spinrwlock_t mu = EASY_SPINRWLOCK_INITIALIZER;
easy_spinrwlock_t mu_list[1000];

double time_for_insert_lockless = 0.0;
double time_for_insert_noresize_lockless = 0.0;
double time_for_insert_at_lockless = 0.0;
double time_for_setvalue_lockless = 0.0;
double time_for_rebucket_lockless = 0.0;
//double time_for_insert_noresize[16] = {0.0};
//double time_for_rebucket[16] = {0.0};


struct eqstr
{
	bool operator()(long  s1, long s2) const
	{
		return (s1 == s2);
	}
};

// A version of each of the hashtable classes we test, that has been
// augumented to provide a common interface.  For instance, the
// sparse_hash_map and dense_hash_map versions set empty-key and
// deleted-key (we can do this because all our tests use int-like
// keys), so the users don't have to.  The hash_map version adds
// resize(), so users can just call resize() for all tests without
// worrying about whether the map-type supports it or not.

template <typename K, typename V, typename H>
class EasyUseSparseHashMap : public sparse_hash_map<K, V, H> {
 public:
  EasyUseSparseHashMap() { this->set_deleted_key(-1); }
};

template <typename K, typename V, typename H>
class EasyUseDenseHashMap : public dense_hash_map<K, V, H> {
 public:
  EasyUseDenseHashMap() {
    this->set_empty_key(-1);
    this->set_deleted_key(-2);
  }
};

// For pointers, we only set the empty key.
template <typename K, typename V, typename H>
class EasyUseSparseHashMap<K*, V, H> : public sparse_hash_map<K*, V, H> {
 public:
  EasyUseSparseHashMap() {}
};

template <typename K, typename V, typename H>
class EasyUseDenseHashMap<K*, V, H> : public dense_hash_map<K*, V, H> {
 public:
  EasyUseDenseHashMap() { this->set_empty_key((K*)(~0)); }
};

template <typename K, typename V, typename H>
class EasyUseHashMap : public unordered_map<K, V, H> {
 public:
  // resize() is called rehash() in tr1
  void resize(size_t r) { this->rehash(r); }
};

template <typename K, typename V>
class EasyUseMap : public map<K, V> {
 public:
  void resize(size_t) {}  // map<> doesn't support resize
};

// Returns the number of hashes that have been done since the last
// call to NumHashesSinceLastCall().  This is shared across all
// HashObject instances, which isn't super-OO, but avoids two issues:
// (1) making HashObject bigger than it ought to be (this is very
// important for our testing), and (2) having to pass around
// HashObject objects everywhere, which is annoying.
static int g_num_hashes;
static int g_num_copies;

int NumHashesSinceLastCall() {
  int retval = g_num_hashes;
  g_num_hashes = 0;
  return retval;
}
int NumCopiesSinceLastCall() {
  int retval = g_num_copies;
  g_num_copies = 0;
  return retval;
}

/*
 * These are the objects we hash.  Size is the size of the object
 * (must be > sizeof(int).  Hashsize is how many of these bytes we
 * use when hashing (must be > sizeof(int) and < Size).
 */
template <int Size, int Hashsize>
class HashObject {
 public:
  typedef HashObject<Size, Hashsize> class_type;
  HashObject() {}
  HashObject(int i) : i_(i) {
    memset(buffer_, i & 255, sizeof(buffer_));  // a "random" char
  }
  HashObject(const HashObject& that) { operator=(that); }
  void operator=(const HashObject& that) {
    g_num_copies++;
    this->i_ = that.i_;
    memcpy(this->buffer_, that.buffer_, sizeof(this->buffer_));
  }

  size_t Hash() const {
    g_num_hashes++;
    int hashval = i_;
    for (size_t i = 0; i < Hashsize - sizeof(i_); ++i) {
      hashval += buffer_[i];
    }
    return std::hash<int>()(hashval);
  }

  bool operator==(const class_type& that) const { return this->i_ == that.i_; }
  bool operator<(const class_type& that) const { return this->i_ < that.i_; }
  bool operator<=(const class_type& that) const { return this->i_ <= that.i_; }

 private:
  int i_;  // the key used for hashing
  char buffer_[Size - sizeof(int)];
};

// A specialization for the case sizeof(buffer_) == 0
template <>
class HashObject<sizeof(int), sizeof(int)> {
 public:
  typedef HashObject<sizeof(int), sizeof(int)> class_type;
  HashObject() {}
  HashObject(int i) : i_(i) {}
  HashObject(const HashObject& that) { operator=(that); }
  void operator=(const HashObject& that) {
    g_num_copies++;
    this->i_ = that.i_;
  }

  size_t Hash() const {
    g_num_hashes++;
    return std::hash<int>()(i_);
  }

  bool operator==(const class_type& that) const { return this->i_ == that.i_; }
  bool operator<(const class_type& that) const { return this->i_ < that.i_; }
  bool operator<=(const class_type& that) const { return this->i_ <= that.i_; }

 private:
  int i_;  // the key used for hashing
};

namespace google {
// Let the hashtable implementations know it can use an optimized memcpy,
// because the compiler defines both the destructor and copy constructor.
template <int Size, int Hashsize>
struct is_relocatable<HashObject<Size, Hashsize>> : std::true_type {};
}

class HashFn {
 public:
  template <int Size, int Hashsize>
  size_t operator()(const HashObject<Size, Hashsize>& obj) const {
    return obj.Hash();
  }
  // Do the identity hash for pointers.
  template <int Size, int Hashsize>
  size_t operator()(const HashObject<Size, Hashsize>* obj) const {
    return reinterpret_cast<uintptr_t>(obj);
  }

  // Less operator for MSVC's hash containers.
  template <int Size, int Hashsize>
  bool operator()(const HashObject<Size, Hashsize>& a,
                  const HashObject<Size, Hashsize>& b) const {
    return a < b;
  }
  template <int Size, int Hashsize>
  bool operator()(const HashObject<Size, Hashsize>* a,
                  const HashObject<Size, Hashsize>* b) const {
    return a < b;
  }
  // These two public members are required by msvc.  4 and 8 are defaults.
  static const size_t bucket_size = 4;
  static const size_t min_buckets = 8;
};

/*
 * Measure resource usage.
 */

class Rusage {
 public:
  /* Start collecting usage */
  Rusage() { Reset(); }

  /* Reset collection */
  void Reset();

  /* Show usage, in nanoseconds */
  double UserTime();

 private:
  steady_clock::time_point start_;
};

inline void Rusage::Reset() { 
  g_num_copies = 0;
  g_num_hashes = 0;  
  start_ = steady_clock::now(); 
}

inline double Rusage::UserTime() {
  auto diff = steady_clock::now() - start_;
  return duration_cast<nanoseconds>(diff).count();
}

static void print_uname() {
#ifdef HAVE_SYS_UTSNAME_H
  struct utsname u;
  if (uname(&u) == 0) {
    printf("%s %s %s %s %s\n", u.sysname, u.nodename, u.release, u.version,
           u.machine);
  }
#endif
}

// Generate stamp for this run
static void stamp_run(int iters, int read_factor) {
  time_t now = time(0);
  printf("======\n");
  fflush(stdout);
  print_uname();
  printf("Average over %d iterations\n", iters);
  printf("read factor = %d\n", read_factor);
  fflush(stdout);
  // don't need asctime_r/gmtime_r: we're not threaded
  printf("Current time (GMT): %s", asctime(gmtime(&now)));
}

// This depends on the malloc implementation for exactly what it does
// -- and thus requires work after the fact to make sense of the
// numbers -- and also is likely thrown off by the memory management
// STL tries to do on its own.

#ifdef HAVE_GOOGLE_MALLOC_EXTENSION_H
#include <google/malloc_extension.h>

static size_t CurrentMemoryUsage() {
  size_t result;
  if (MallocExtension::instance()->GetNumericProperty(
          "generic.current_allocated_bytes", &result)) {
    return result;
  } else {
    return 0;
  }
}

#else /* not HAVE_GOOGLE_MALLOC_EXTENSION_H */
static size_t CurrentMemoryUsage() { return 0; }

#endif

static void report(char const* title, double t, int iters, size_t start_memory,
                   size_t end_memory) {
  // Construct heap growth report text if applicable
  char heap[100] = "";
  if (end_memory > start_memory) {
    snprintf(heap, sizeof(heap), "%7.1f MB",
             (end_memory - start_memory) / 1048576.0);
  }

  printf("%-20s %6.1f ns  (%8d hashes, %8d copies)%s\n", title, (t / iters),
         NumHashesSinceLastCall(), NumCopiesSinceLastCall(), heap);
  fflush(stdout);
}

template <class MapType>
static void time_map_grow(int iters) {
  MapType set;
  Rusage t;

  const size_t start = CurrentMemoryUsage();
  t.Reset();
  for (int i = 0; i < iters; i++) {
    set[i] = i + 1;
  }
  double ut = t.UserTime();
  const size_t finish = CurrentMemoryUsage();
  report("map_grow", ut, iters, start, finish);
}

template <class MapType>
static void time_map_grow_predicted(int iters) {
  MapType set;
  Rusage t;

  const size_t start = CurrentMemoryUsage();
  set.resize(iters);
  t.Reset();
  for (int i = 0; i < iters; i++) {
    set[i] = i + 1;
  }
  double ut = t.UserTime();
  const size_t finish = CurrentMemoryUsage();
  report("map_predict/grow", ut, iters, start, finish);
}

template <class MapType>
static void time_map_replace(int iters) {
  MapType set;
  Rusage t;
  int i;

  for (i = 0; i < iters; i++) {
    set[i] = i + 1;
  }

  t.Reset();
  for (i = 0; i < iters; i++) {
    set[i] = i + 1;
  }
  double ut = t.UserTime();

  report("map_replace", ut, iters, 0, 0);
}

template <class MapType>
static void time_map_fetch(int iters, const vector<int>& indices,
                           char const* title) {
  MapType set;
  Rusage t;
  int r;
  int i;

  for (i = 0; i < iters; i++) {
    set[i] = i + 1;
  }

  r = 1;
  t.Reset();
  for (i = 0; i < iters; i++) {
    r ^= static_cast<int>(set.find(indices[i]) != set.end());
  }
  double ut = t.UserTime();

  srand(r);  // keep compiler from optimizing away r (we never call rand())
  report(title, ut, iters, 0, 0);
}

template <class MapType>
static void time_map_fetch_sequential(int iters) {
  vector<int> v(iters);
  for (int i = 0; i < iters; i++) {
    v[i] = i;
  }
  time_map_fetch<MapType>(iters, v, "map_fetch_sequential");
}

// Apply a pseudorandom permutation to the given vector.
static void shuffle(vector<int>* v) {
  srand(9);
  for (int n = v->size(); n >= 2; n--) {
    swap((*v)[n - 1], (*v)[static_cast<unsigned>(rand()) % n]);
  }
}

template <class MapType>
static void time_map_fetch_random(int iters) {
  vector<int> v(iters);
  for (int i = 0; i < iters; i++) {
    v[i] = i;
  }
  shuffle(&v);
  time_map_fetch<MapType>(iters, v, "map_fetch_random");
}

template <class MapType>
static void time_map_fetch_empty(int iters) {
  MapType set;
  Rusage t;
  int r;
  int i;

  r = 1;
  t.Reset();
  for (i = 0; i < iters; i++) {
    r ^= static_cast<int>(set.find(i) != set.end());
  }
  double ut = t.UserTime();

  srand(r);  // keep compiler from optimizing away r (we never call rand())
  report("map_fetch_empty", ut, iters, 0, 0);
}

template <class MapType>
static void time_map_remove(int iters) {
  MapType set;
  Rusage t;
  int i;

  for (i = 0; i < iters; i++) {
    set[i] = i + 1;
  }

  t.Reset();
  for (i = 0; i < iters; i++) {
    set.erase(i);
  }
  double ut = t.UserTime();

  report("map_remove", ut, iters, 0, 0);
}

template <class MapType>
static void time_map_toggle(int iters) {
  MapType set;
  Rusage t;
  int i;

  const size_t start = CurrentMemoryUsage();
  t.Reset();
  for (i = 0; i < iters; i++) {
    set[i] = i + 1;
    set.erase(i);
  }

  double ut = t.UserTime();
  const size_t finish = CurrentMemoryUsage();

  report("map_toggle", ut, iters, start, finish);
}

template <class MapType>
static void time_map_iterate(int iters) {
  MapType set;
  Rusage t;
  int r;
  int i;

  for (i = 0; i < iters; i++) {
    set[i] = i + 1;
  }

  r = 1;
  t.Reset();
  for (typename MapType::const_iterator it = set.begin(), it_end = set.end();
       it != it_end; ++it) {
    r ^= it->second;
  }

  double ut = t.UserTime();

  srand(r);  // keep compiler from optimizing away r (we never call rand())
  report("map_iterate", ut, iters, 0, 0);
}

template <class MapType>
static void stresshashfunction(int desired_insertions, int map_size,
                               int stride) {
  Rusage t;
  int num_insertions = 0;
  // One measurement of user time (in nanoseconds) is done for each iteration of
  // the outer loop.  The times are summed.
  double total_nanoseconds = 0;
  const int k = desired_insertions / map_size;
  MapType set;
  for (int o = 0; o < k; o++) {
    set.clear();
    set.resize(map_size);
    t.Reset();
    const int maxint = (1ull << (sizeof(int) * 8 - 1)) - 1;
    // Use n arithmetic sequences.  Using just one may lead to overflow
    // if stride * map_size > maxint.  Compute n by requiring
    // stride * map_size/n < maxint, i.e., map_size/(maxint/stride) < n
    char* key;  // something we can do math on
    const int n = map_size / (maxint / stride) + 1;
    for (int i = 0; i < n; i++) {
      key = NULL;
      key += i;
      for (int j = 0; j < map_size / n; j++) {
        key += stride;
        set[reinterpret_cast<typename MapType::key_type>(key)] =
            ++num_insertions;
      }
    }
    total_nanoseconds += t.UserTime();
  }
  printf("stresshashfunction map_size=%d stride=%d: %.1fns/insertion\n",
         map_size, stride, total_nanoseconds / num_insertions);
}

template <class MapType>
static void stresshashfunction(int num_inserts) {
  static const int kMapSizes[] = {256, 1024};
  for (unsigned i = 0; i < sizeof(kMapSizes) / sizeof(kMapSizes[0]); i++) {
    const int map_size = kMapSizes[i];
    for (int stride = 1; stride <= map_size; stride *= map_size) {
      stresshashfunction<MapType>(num_inserts, map_size, stride);
    }
  }
}

template <class MapType, class StressMapType>
static void measure_map(const char* label, int obj_size, int iters,
                        bool stress_hash_function) {
  printf("\n%s (%d byte objects, %d iterations):\n", label, obj_size, iters);
  if (1) time_map_grow<MapType>(iters);
  if (1) time_map_grow_predicted<MapType>(iters);
  if (1) time_map_replace<MapType>(iters);
  if (1) time_map_fetch_random<MapType>(iters);
  if (1) time_map_fetch_sequential<MapType>(iters);
  if (1) time_map_fetch_empty<MapType>(iters);
  if (1) time_map_remove<MapType>(iters);
  if (1) time_map_toggle<MapType>(iters);
  if (1) time_map_iterate<MapType>(iters);
  // This last test is useful only if the map type uses hashing.
  // And it's slow, so use fewer iterations.
  if (stress_hash_function) {
    // Blank line in the output makes clear that what follows isn't part of the
    // table of results that we just printed.
    puts("");
    stresshashfunction<StressMapType>(iters / 4);
  }
}

template <class ObjType>
static void test_all_maps(int obj_size, int iters) {
  const bool stress_hash_function = obj_size <= 8;

  if (FLAGS_test_sparse_hash_map)
    measure_map<EasyUseSparseHashMap<ObjType, int, HashFn>,
                EasyUseSparseHashMap<ObjType*, int, HashFn>>(
        "SPARSE_HASH_MAP", obj_size, iters, stress_hash_function);

  if (FLAGS_test_dense_hash_map)
    measure_map<EasyUseDenseHashMap<ObjType, int, HashFn>,
                EasyUseDenseHashMap<ObjType*, int, HashFn>>(
        "DENSE_HASH_MAP", obj_size, iters, stress_hash_function);

  if (FLAGS_test_hash_map)
    measure_map<EasyUseHashMap<ObjType, int, HashFn>,
                EasyUseHashMap<ObjType*, int, HashFn>>(
        "STANDARD HASH_MAP", obj_size, iters, stress_hash_function);

  if (FLAGS_test_map)
    measure_map<EasyUseMap<ObjType, int>, EasyUseMap<ObjType*, int>>(
        "STANDARD MAP", obj_size, iters, false);
}

void thread_lookup(dense_hash_map_lockless<long, long, hash<long>, eqstr>& ht, int iter, int offset, int threadnum){
  for (long j = offset*iter/threadnum; j < (offset+1)*iter/threadnum; j++) {
    ht.find_wait_free(j);
  }
}

void thread_lookup_rwlock(dense_hash_map<long, long, hash<long>, eqstr>& ht, int iter, int offset, int threadnum){
  for (long j = offset*iter/threadnum; j < (offset+1)*iter/threadnum; j++) {
    spin_rd_lock l(mu);
    ht.find(j);
  }
}

void thread_lookup_rwlock_and_shaders(dense_hash_map<long, long, hash<long>, eqstr>* ht, int iter, int offset, int threadnum){
  for (long j = offset*iter/threadnum; j < (offset+1)*iter/threadnum; j++){
    int bucket = j % 1000;
    spin_rd_lock l(mu_list[bucket]);
    ht[bucket].find(j);
  }
}

void thread_insert(dense_hash_map_lockless<long, long, hash<long>, eqstr>& ht, int iter, int offset, int threadnum){
  for (long j = offset*iter/threadnum; j < (offset+1)*iter/threadnum; j++) {
    ht.insert_lockless(std::move(std::pair<long, long>(j, j+10)));
  }
}

void thread_insert_rwlock(dense_hash_map<long, long, hash<long>, eqstr>& ht, int iter, int offset, int threadnum){
  for (long j = offset*iter/threadnum; j < (offset+1)*iter/threadnum; j++) {
    spin_wr_lock l(mu);
    ht.insert(std::move(std::pair<long, long>(j, j+10)));
  }
}

void thread_insert_rwlock_and_shaders(dense_hash_map<long, long, hash<long>, eqstr>* ht, int iter, int offset, int threadnum){
  for (long j = offset*iter/threadnum; j < (offset+1)*iter/threadnum; j++) {
    int bucket = j % 1000;
    spin_wr_lock l(mu_list[bucket]);
    ht[bucket].insert(std::move(std::pair<long, long>(j, j+10)));
 }
}

void thread_find_insert(dense_hash_map_lockless<long, long, hash<long>, eqstr>& ht, int iter, int offset, int threadnum){
  for  (long j = offset*iter/threadnum; j < (offset+1)*iter/threadnum; j++) {
    ht.insert_lockless(std::move(std::pair<long, long>(j, j+10)));
    ht.find_wait_free(j);
  }
}

void thread_erase(dense_hash_map_lockless<long, long, hash<long>, eqstr>& ht, int iter, int offset, int threadnum){
  for (long j = offset*iter/threadnum; j < (offset+1)*iter/threadnum; j++) {
    ht.erase_lockless(j);
  }
}

void test_parallel_find(int threadnum, int iter){
  dense_hash_map_lockless< long,  long, hash<long>, eqstr> ht;
  ht.set_empty_key_and_value(-1, 2147483647);
  ht.set_deleted_key(-2);
  ht.set_counternum(16);
  
  for (long i = 0; i < iter; i++)
    ht.insert_lockless(std::move(std::pair<long ,long>(i, i + 10)));
  std::vector<std::thread> insert_threads(threadnum);
  auto st_time = std::chrono::high_resolution_clock::now();
  for (long i = 0 ; i < threadnum; i++) {
    insert_threads[i] = std::thread(thread_lookup, std::ref(ht), iter, i, threadnum);
  }
  for (auto &t : insert_threads) {
    t.join();
  }
  auto ed_time = std::chrono::high_resolution_clock::now();
  auto seconds_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(ed_time - st_time).count();
  std::cout<<"parallel_find\t"<<"Time: "<<seconds_elapsed<<"\t"<<"Throughput: "<<iter/seconds_elapsed<<std::endl;
}

void test_parallel_insert(int threadnum, int iter){
  dense_hash_map_lockless< long,  long, hash<long>, eqstr> ht;
  ht.set_empty_key_and_value(-1, 2147483647);
  ht.set_deleted_key(-2);
  ht.set_counternum(16);
  std::vector<std::thread> insert_threads(threadnum);
  auto st_time = std::chrono::high_resolution_clock::now();
  for (long i = 0 ; i < threadnum; i++) {
    insert_threads[i] = std::thread(thread_insert, std::ref(ht), iter, i, threadnum);
  }
  for (auto &t : insert_threads) {
    t.join();
  }
  auto ed_time = std::chrono::high_resolution_clock::now();
  auto seconds_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(ed_time - st_time).count();
  std::cout<<"parallel_insert\t"<<"Time: "<<seconds_elapsed<<"\t"<<"Throughput: "<<iter/seconds_elapsed<<std::endl;
}

void test_parallel_find_and_insert(int threadnum, int iter){
  dense_hash_map_lockless< long,  long, hash<long>, eqstr> ht;
  ht.set_empty_key_and_value(-1, 2147483647);
  ht.set_deleted_key(-2);
  ht.set_counternum(16);
  std::vector<std::thread> insert_threads(threadnum);
  auto st_time = std::chrono::high_resolution_clock::now();
  for (long i = 0 ; i < threadnum; i++){
    insert_threads[i] = std::thread(thread_find_insert, std::ref(ht), iter, i, threadnum);
  }
  for (auto &t : insert_threads) {
    t.join();
  }
  auto ed_time = std::chrono::high_resolution_clock::now();
  auto seconds_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(ed_time - st_time).count();
  std::cout<<"parallel_find_and_insert\t"<<"Time: "<<seconds_elapsed<<"\t"<<"Throughput: "<<2*iter/seconds_elapsed<<std::endl;
}

void test_parallel_erase(int threadnum, int iter){
  dense_hash_map_lockless< long,  long, hash<long>, eqstr> ht;
  ht.set_empty_key_and_value(-1, 2147483647);
  ht.set_deleted_key(-2);
  ht.set_counternum(16);
  for (long i = 0; i < iter; i++)
    ht.insert(std::move(std::pair<long ,long>(i, i + 10)));
  std::vector<std::thread> insert_threads(threadnum);
  auto st_time = std::chrono::high_resolution_clock::now();
  for (long i = 0 ; i < threadnum; i++) {
    insert_threads[i] = std::thread(thread_erase, std::ref(ht), iter, i, threadnum);
  }
  for (auto &t : insert_threads) {
    t.join();
  }
  auto ed_time = std::chrono::high_resolution_clock::now();
  auto seconds_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(ed_time - st_time).count();
  std::cout<<"parallel_erase\t"<<"Time: "<<seconds_elapsed<<"\t"<<"Throughput: "<<iter/seconds_elapsed<<std::endl;
}

int lookup(dense_hash_map_lockless<long, long, hash<long>, eqstr>& ht, long key){
  auto it = ht.find(key);
  if (it == ht.end()) {
    return 0;
  }else{
    return 1;
  }
}

int lookup_rwlock(dense_hash_map<long, long, hash<long>, eqstr>& ht, long key){
  spin_rd_lock l(mu);
  auto it = ht.find(key);
  if(it == ht.end()){
    return 0;
  }else{
    return 1;
 }
}

int lookup_rwlock_and_shaders(dense_hash_map<long, long, hash<long>, eqstr>* ht, long key){
  int bucket = key % 1000;
  spin_rd_lock l(mu_list[bucket]);
	auto it = ht[bucket].find(key);
	if(it == ht[bucket].end()){
		return 0;
	}else{
		return 1;
	}
}

void find_or_insert(dense_hash_map_lockless<long, long, hash<long>, eqstr>& ht,long *keys, long ReadLoops, int i){
  for (long j = 0; j < ReadLoops; j++) {
    if (!lookup(std::ref(ht), keys[j])) {
      auto it1 = ht.insert_lockless(std::move(std::pair<long, long >(keys[j], keys[j]+10)));
    }
  }
}

void find_or_insert_rwlock(dense_hash_map<long, long, hash<long>, eqstr>& ht,long *keys, long ReadLoops, int i){
  for(long j = 0; j < ReadLoops; j++){
    if (!lookup_rwlock(std::ref(ht), keys[j])) {
      spin_wr_lock l(mu);
      auto it1 = ht.insert(std::move(std::pair<long, long >(keys[j], keys[j]+10)));
    }
  }
}

void find_or_insert_rwlock_and_shaders(dense_hash_map<long, long, hash<long>, eqstr>* ht,long *keys, long ReadLoops, int i){
  for(long j = 0; j < ReadLoops; j++){
    if(!lookup_rwlock_and_shaders(ht, keys[j])){
      int bucket = keys[j] % 1000;
      spin_wr_lock l(mu_list[bucket]);
      auto it1 = ht[bucket].insert(std::move(std::pair<long, long >(keys[j], keys[j]+10)));
    }
  }
}

void find_or_insert_with_shaders(dense_hash_map_lockless<long, long, hash<long>, eqstr>* ht,long *keys, long ReadLoops, int i){
  for(long j = 0; j < ReadLoops; j++){
    if (!lookup(std::ref(ht[0]), keys[j])) {
      auto it1 = ht[0].insert_lockless(std::move(std::pair<long, long >(keys[j], keys[j]+10)));
    }
  }
}

void gen_thd(std::set<long>* segs, int thdid, long seg_num) {
  int i = 0;
  while(true) {
    if (segs->size() == seg_num) 
      break;
    
    long gen_key = (rand() % ( 10 * seg_num)) * (1 + thdid);
    if (segs->find(gen_key) == segs->end()) {
      segs->insert(gen_key);
    }
  }
}

void insert_thd2(dense_hash_map_lockless<long, long, hash<long>, eqstr>& ht, const std::set<long>& keys , int thdid) {
  auto it = keys.begin();
  for (; it !=keys.end(); it++) {
    long id = *it;
    if(!lookup(std::ref(ht), id)){
      auto it1 = ht.insert_lockless(std::move(std::pair<long, long >(id, id+10)));
    }
  } 
}

void gen_hybrid_thd(std::set<long>* segs, std::vector<long>* insert_keys, std::vector<long>* lookup_keys, int thdid, long seg_num) {
  {
    auto it = segs->begin();
    for (; it != segs->end(); it++) {
      lookup_keys->push_back(*it);
    }
  }
  int i = 0;
  while(true) {
    if (segs->size() == seg_num) 
      break;
    
    long gen_key = (rand() % ( 1024 * seg_num) + 1) * (1 + thdid);
    if (segs->find(gen_key) == segs->end()) {
      segs->insert(gen_key);
      insert_keys->push_back(gen_key);
    }
  }
}

void hybrid_thd_f(dense_hash_map_lockless<long, long, hash<long>, eqstr>& ht, std::vector<long>* insert_keys, std::vector<long>* lookup_keys,  long ops, int thdid) {
  auto insert_it = insert_keys->begin();
  auto lookup_it = lookup_keys->begin(); 
 
  for (long i = 0; i < ops ; i++) {
    int k = i % 2; 
    // 80% lookup
    if ( k == 0 ) {
      if (lookup_it == lookup_keys->end()) lookup_it = lookup_keys->begin();
      if (!lookup(std::ref(ht), *lookup_it)) {
        auto it1 = ht.insert_lockless(std::move(std::pair<long, long >(*lookup_it, *lookup_it+10)));
        }
        lookup_it++;
      } else if (k == 1) {
        // 20% insert
        if (insert_it == insert_keys->end()) insert_it = insert_keys->begin();
        if(!lookup(std::ref(ht), *lookup_it)){
        auto it1 = ht.insert_lockless(std::move(std::pair<long, long >(*lookup_it, *lookup_it+10)));
        }
        insert_it++;
      } 
  } 
}

void test_parallel_hybrid(int thread_num, int iter, int read_factor){
  dense_hash_map_lockless<long, long, hash<long>, eqstr> ht;
  ht.set_empty_key_and_value(-1, 2147483647);
  ht.set_deleted_key(-2);
  ht.set_counternum(16);
  std::vector<long> lookup_keys;
  std::vector<std::set<long>> seg_keys(thread_num);
  long seg_num = read_factor * iter / thread_num;
  std::vector<std::thread> gen_threads(thread_num);
  for (int i = 0; i < thread_num; i++) {
    gen_threads[i] = std::thread(gen_thd, &seg_keys[i], i, seg_num);
  }

  for (auto &t : gen_threads) {
    t.join();
  }

  std::vector<std::thread> insert_threads(thread_num);

  for (size_t i = 0; i < thread_num; ++i) {
    insert_threads[i] = std::thread(insert_thd2, std::ref(ht), seg_keys[i], i);
  }
  for (auto &t : insert_threads) {
    t.join();
  }

  {
    std::vector<std::vector<long> > lookup_keys(thread_num);
    std::vector<std::vector<long> > insert_keys(thread_num);

    std::vector<std::thread> gen_threads(thread_num);
    for (int i = 0; i < thread_num; i++) {
      gen_threads[i] =
      std::thread(gen_hybrid_thd, &seg_keys[i], &insert_keys[i], &lookup_keys[i], i, 2 * seg_num);
    }

    for (auto &t : gen_threads) {
      t.join();
    }
    std::vector<std::thread> hybrid_threads(thread_num);
    auto st_time = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < thread_num; ++i) {
        hybrid_threads[i] =
          std::thread(hybrid_thd_f, std::ref(ht), &(insert_keys[i]), &(lookup_keys[i]), read_factor * iter / thread_num, i);
    }
    for (auto &t : hybrid_threads) {
      t.join();
    }
    auto ed_time = std::chrono::high_resolution_clock::now();
    auto seconds_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(ed_time - st_time).count();
    std::cout<<"parallel_hybrid\t"<<"Time: "<<seconds_elapsed<<"\t"<<"Throughput: "<<read_factor*iter/seconds_elapsed<<std::endl;
  }

}

void test_parallel_8find_and_2insert_hotspot(int threadnum, int iter, int read_factor){
  dense_hash_map_lockless< long,  long, hash<long>, eqstr> ht;
  ht.set_empty_key_and_value(-1, 2147483647);
  ht.set_deleted_key(-2);
  ht.set_counternum(16);
  bool* flag = (bool *)malloc(sizeof(bool)*2147483647);
  srand((unsigned)time(NULL)); 
  long *keys = (long *)malloc(sizeof(long)*iter);
  long *counter = (long *)malloc(sizeof(long)*iter);
  long *hotkeys, *coldkeys;
  hotkeys = keys;
  coldkeys = keys + iter/5;
  for (long i = 0; i < 2147483647; i++) {
    flag[i] = 0;
  }
  for (long i = 0; i < iter; i++) {
    counter[i] = 1;
  }
  int index = 0;
  while (index < iter) {
    long j = rand() % 2147483647;
    if (flag[j] == 1) // the number is already set as a key
      continue;
    else{ // the number is not selected as a key
      keys[index] = j;
      index++;
      flag[j] = 1;
    }
  }
  free(flag);
  long** lookup_keys = (long **)malloc(sizeof(long *) * threadnum);

  for (size_t i = 0; i < threadnum; i++)
    lookup_keys[i] = (long *)malloc(sizeof(long) * read_factor * iter/threadnum);

  for (long k = 0; k < threadnum; k ++) {
    for (long i = 0; i < read_factor * iter/threadnum; i++) {
      long j = rand()%10;
      if (j < 8) {
        long pos = rand()%(iter/5);
	lookup_keys[k][i] = hotkeys[pos];
      } else {
        long pos = rand()%(iter * 4 / 5);
	lookup_keys[k][i] = coldkeys[pos];
      }
    }
  }
  std::vector<std::thread> lookup_threads(threadnum);
  auto st_time = std::chrono::high_resolution_clock::now();
  for (size_t i =0 ; i < threadnum; i++) {
    lookup_threads[i] = std::thread(find_or_insert, std::ref(ht), lookup_keys[i], read_factor*iter/threadnum, i);
  }
  for(auto &t : lookup_threads) {
    t.join();
  }
  auto ed_time = std::chrono::high_resolution_clock::now();
  auto seconds_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(ed_time - st_time).count();
  std::cout<<"parallel_find_or_create\t"<<"Time: "<<seconds_elapsed<<"\t"<<"Throughput: "<<read_factor*iter/seconds_elapsed<<std::endl;
}

void test_parallel_find_or_create_with_shaders(int threadnum, int iter, int read_factor){
  dense_hash_map_lockless< long,  long, hash<long>, eqstr> ht[1000];
  for (int i = 0; i < 1000 ; i++){
    ht[i].set_empty_key_and_value(-1, 2147483647);
    ht[i].set_deleted_key(-2);
    ht[i].set_counternum(16);
  }
  bool* flag = (bool *)malloc(sizeof(bool)*2147483647);
  srand((unsigned)time(NULL)); 
  long *keys = (long *)malloc(sizeof(long)*iter);
  long *counter = (long *)malloc(sizeof(long)*iter);
  long *hotkeys, *coldkeys;
  hotkeys = keys;
  coldkeys = keys + iter/5;
  for(long i = 0; i < 2147483647; i++){
    flag[i] = 0;
  }
  for(long i = 0; i < iter; i++){
    counter[i] = 1;
  }
  int index = 0;
  while (index < iter){
    long j = rand() % 2147483647;
    if(flag[j] == 1) // the number is already set as a key
      continue;
    else{ // the number is not selected as a key
      keys[index] = j;
      index++;
      flag[j] = 1;
   }
  }
  free(flag);
  long** lookup_keys = (long **)malloc(sizeof(long *) * threadnum);

  for (size_t i = 0; i < threadnum; i++)
    lookup_keys[i] = (long *)malloc(sizeof(long) * read_factor *iter/threadnum);

  for (long k = 0; k < threadnum; k ++) {
    for (long i = 0; i < read_factor*iter/threadnum; i++) {
      long j = rand()%10;
      if (j < 8) {
        long pos = rand()%(iter/5);
	lookup_keys[k][i] = hotkeys[pos];
      }else{
	long pos = rand()%(iter * 4 / 5);
	lookup_keys[k][i] = coldkeys[pos];
      }
    }
  }
  std::vector<std::thread> lookup_threads(threadnum);
  auto st_time = std::chrono::high_resolution_clock::now();
  for (size_t i = 0 ; i < threadnum; i++) {
    lookup_threads[i] = std::thread(find_or_insert_with_shaders, ht, lookup_keys[i], read_factor*iter/threadnum, i);
  }
  for (auto &t : lookup_threads) {
    t.join();
  }
  auto ed_time = std::chrono::high_resolution_clock::now();
  auto seconds_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(ed_time - st_time).count();
  std::cout<<"parallel_find_or_create_with_shaders\t"<<"Time: "<<seconds_elapsed<<"\t"<<"Throughput: "<<read_factor*iter/seconds_elapsed<<std::endl;
}


void test_dense_hash_map_parallel(int threadnum, int iter, int read_factor){
  test_parallel_find(threadnum, iter);
  test_parallel_insert(threadnum, iter);
  test_parallel_hybrid(threadnum, iter, read_factor);
}


void test_parallel_find_rwlock(int threadnum, int iter){
  dense_hash_map< long,  long, hash<long>, eqstr> ht;
  ht.set_empty_key(-1);
  ht.set_deleted_key(-2);
  for (long i = 0; i < iter; i++)
    ht.insert(std::move(std::pair<long ,long>(i, i + 10)));
  std::vector<std::thread> insert_threads(threadnum);
  auto st_time = std::chrono::high_resolution_clock::now();
  for (long i = 0 ; i < threadnum; i++) {
    insert_threads[i] = std::thread(thread_lookup_rwlock, std::ref(ht), iter, i, threadnum);
  }
  for (auto &t : insert_threads) {
    t.join();
  }
  auto ed_time = std::chrono::high_resolution_clock::now();
  auto seconds_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(ed_time - st_time).count();
  std::cout<<"parallel_find\t"<<"Time: "<<seconds_elapsed<<"\t"<<"Throughput: "<<iter/seconds_elapsed<<std::endl;
}

void test_parallel_insert_rwlock(int threadnum, int iter){
  dense_hash_map< long,  long, hash<long>, eqstr> ht;
  ht.set_empty_key(-1);
  ht.set_deleted_key(-2);
  std::vector<std::thread> insert_threads(threadnum);
  auto st_time = std::chrono::high_resolution_clock::now();
  for (long i = 0 ; i < threadnum; i++) {
    insert_threads[i] = std::thread(thread_insert_rwlock, std::ref(ht), iter, i, threadnum);
  }
  for (auto &t : insert_threads) {
    t.join();
  }
  auto ed_time = std::chrono::high_resolution_clock::now();
  auto seconds_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(ed_time - st_time).count(); 
  std::cout<<"parallel_insert\t"<<"Time: "<<seconds_elapsed<<"\t"<<"Throughput: "<<iter/seconds_elapsed<<std::endl;
}

void test_parallel_8find_and_2insert_hotspot_rwlock(int threadnum, int iter){
  dense_hash_map< long,  long, hash<long>, eqstr> ht;
  ht.set_empty_key(-1);
  ht.set_deleted_key(-2);
  bool* flag = (bool *)malloc(sizeof(bool)*2147483647);
  srand((unsigned)time(NULL)); 
  long *keys = (long *)malloc(sizeof(long)*iter);
  long *counter = (long *)malloc(sizeof(long)*iter);
  long *hotkeys, *coldkeys;
  hotkeys = keys;
  coldkeys = keys + iter/5;
  for (long i = 0; i < 2147483647; i++) {
    flag[i] = 0;
  }
  for (long i = 0; i < iter; i++) {
    counter[i] = 1;
  }
  int index = 0;
  while (index < iter) {
    long j = rand() % 2147483647;
    if (flag[j] == 1) // the number is already set as a key
      continue;
    else{ // the number is not selected as a key
      keys[index] = j;
      index++;
      flag[j] = 1;
    }
  }
  free(flag);
  long** lookup_keys = (long **)malloc(sizeof(long *) * threadnum);

  for (size_t i = 0; i < threadnum; i++)
    lookup_keys[i] = (long *)malloc(sizeof(long) * 5*iter/threadnum);

  for (long k = 0; k < threadnum; k ++) {
    for (long i = 0; i < 5*iter/threadnum; i++){
      long j = rand()%10;
      if (j < 8) {
        long pos = rand()%(iter/5);
	lookup_keys[k][i] = hotkeys[pos];
      }else {
        long pos = rand()%(iter * 4 / 5);
	lookup_keys[k][i] = coldkeys[pos];
      }
    }
  }
  std::vector<std::thread> lookup_threads(threadnum);
  auto st_time = std::chrono::high_resolution_clock::now();
  for (size_t i = 0 ; i < threadnum; i++) {
    lookup_threads[i] = std::thread(find_or_insert_rwlock, std::ref(ht), lookup_keys[i], 5*iter/threadnum, i);
  }
  for (auto &t : lookup_threads) {
    t.join();
  }
  auto ed_time = std::chrono::high_resolution_clock::now();
  auto seconds_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(ed_time - st_time).count();
  std::cout<<"parallel_find_or_create\t"<<"Time: "<<seconds_elapsed<<"\t"<<"Throughput: "<<5*iter/seconds_elapsed<<std::endl;
}

void test_dense_hash_map_with_rwlock(int threadnum, int iter){
  test_parallel_find_rwlock(threadnum, iter);
  test_parallel_insert_rwlock(threadnum, iter);
}

void test_parallel_find_rwlock_and_shaders(int threadnum, int iter){
  dense_hash_map< long,  long, hash<long>, eqstr> ht[1000];
  for (int i = 0; i < 1000 ; i++){
    ht[i].set_empty_key(-1);
    ht[i].set_deleted_key(-2);
  }
  for (long i = 0; i < iter; i++) {
    int bucket = i % 1000;
    ht[bucket].insert(std::move(std::pair<long ,long>(i, i + 10)));
  }
  std::vector<std::thread> insert_threads(threadnum);
  auto st_time = std::chrono::high_resolution_clock::now();
  for(long i =0 ; i < threadnum; i++){
    insert_threads[i] = std::thread(thread_lookup_rwlock_and_shaders, ht, iter, i, threadnum);
  }
  for(auto &t : insert_threads){
    t.join();
  }
  auto ed_time = std::chrono::high_resolution_clock::now();
  auto seconds_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(ed_time - st_time).count();
  std::cout<<"parallel_find\t"<<"Time: "<<seconds_elapsed<<"\t"<<"Throughput: "<<iter/seconds_elapsed<<std::endl;
}

void test_parallel_insert_rwlock_and_shaders(int threadnum, int iter){
  dense_hash_map< long,  long, hash<long>, eqstr> ht[1000];
  for (int i = 0; i < 1000 ; i++) {
    ht[i].set_empty_key(-1);
    ht[i].set_deleted_key(-2);
  }
  std::vector<std::thread> insert_threads(threadnum);
  auto st_time = std::chrono::high_resolution_clock::now();
  for (long i = 0 ; i < threadnum; i++) {
    insert_threads[i] = std::thread(thread_insert_rwlock_and_shaders, ht, iter, i, threadnum);
  }
  for(auto &t : insert_threads){
    t.join();
  }
  auto ed_time = std::chrono::high_resolution_clock::now();
  auto seconds_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(ed_time - st_time).count();
  std::cout<<"parallel_insert\t"<<"Time: "<<seconds_elapsed<<"\t"<<"Throughput: "<<iter/seconds_elapsed<<std::endl;
}

void test_parallel_8find_and_2insert_hotspot_rwlock_and_shaders(int threadnum, int iter, int read_factor){
  dense_hash_map< long,  long, hash<long>, eqstr> ht[1000];
  for (int i = 0; i < 1000 ; i++) {
    ht[i].set_empty_key(-1);
    ht[i].set_deleted_key(-2);
  }
  bool* flag = (bool *)malloc(sizeof(bool)*2147483647);
  srand((unsigned)time(NULL)); 
  long *keys = (long *)malloc(sizeof(long)*iter);
  long *counter = (long *)malloc(sizeof(long)*iter);
  long *hotkeys, *coldkeys;
  hotkeys = keys;
  coldkeys = keys + iter/5;
  for (long i = 0; i < 2147483647; i++) {
    flag[i] = 0;
  }
  for (long i = 0; i < iter; i++) {
    counter[i] = 1;
  }
  int index = 0;
  while (index < iter) {
    long j = rand() % 2147483647;
    if (flag[j] == 1) // the number is already set as a key
      continue;
    else { // the number is not selected as a key
     keys[index] = j;
     index++;
     flag[j] = 1;
    }
  }
  free(flag);
  long** lookup_keys = (long **)malloc(sizeof(long *) * threadnum);

  for (size_t i = 0; i < threadnum; i++)
  lookup_keys[i] = (long *)malloc(sizeof(long) * read_factor * iter/threadnum);
  for (long k = 0; k < threadnum; k++) {
    for (long i = 0; i < read_factor*iter/threadnum; i++) {
      long j = rand()%10;
      if (j < 8) {
        long pos = rand()%(iter/5);
        lookup_keys[k][i] = hotkeys[pos];
      }else{
        long pos = rand()%(iter * 4 / 5);
        lookup_keys[k][i] = coldkeys[pos];
      }
    }
  }
  std::vector<std::thread> lookup_threads(threadnum);
  auto st_time = std::chrono::high_resolution_clock::now();
  for (size_t i = 0 ; i < threadnum; i++) {
    lookup_threads[i] = std::thread(find_or_insert_rwlock_and_shaders, ht, lookup_keys[i], read_factor*iter/threadnum, i);
  }
  for (auto &t : lookup_threads) {
    t.join();
  }
  auto ed_time = std::chrono::high_resolution_clock::now();
  auto seconds_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(ed_time - st_time).count();
  std::cout<<"parallel_find_or_create\t"<<"Time: "<<seconds_elapsed<<"\t"<<"Throughput: "<<read_factor*iter/seconds_elapsed<<std::endl;

}

void insert_thd2_rwlock_and_shaders(dense_hash_map<long, long, hash<long>, eqstr>* ht, const std::set<long>& keys , int thdid) {
  auto it = keys.begin();
  for (; it !=keys.end(); it++) {
    long id = *it;
    if (!lookup_rwlock_and_shaders(ht, id)) {
      int bucket = id % 1000;
      spin_wr_lock l(mu_list[bucket]);
      auto it1 = ht[bucket].insert(std::move(std::pair<long, long >(id, id+10)));
    }
  } 
}


void hybrid_thd_f_rwlock_and_shaders(dense_hash_map<long, long, hash<long>, eqstr>* ht, std::vector<long>* insert_keys, std::vector<long>* lookup_keys,  long ops, int thdid) {
  auto insert_it = insert_keys->begin();
  auto lookup_it = lookup_keys->begin(); 
 
  for (long i = 0; i < ops ; i++) {

    int k = i % 2;
    
      // 80% lookup
    if ( k == 0 ) {
      if (lookup_it == lookup_keys->end()) lookup_it = lookup_keys->begin();
      if(!lookup_rwlock_and_shaders(ht, *lookup_it)){
        int bucket = *lookup_it % 1000;
        spin_wr_lock l(mu_list[bucket]);
        auto it1 = ht[bucket].insert(std::move(std::pair<long, long >(*lookup_it, *lookup_it+10)));
      }
      lookup_it++;
      } else if (k == 1) {
        // 20% insert
        if (insert_it == insert_keys->end()) insert_it = insert_keys->begin();
        if(!lookup_rwlock_and_shaders(ht, *lookup_it)){
          int bucket = *lookup_it % 1000;
          spin_wr_lock l(mu_list[bucket]);
          auto it1 = ht[bucket].insert(std::move(std::pair<long, long >(*lookup_it, *lookup_it+10)));
        }
        insert_it++;
      }
  } 
}

void test_parallel_hybrid_rwlock_and_shaders(int thread_num, int iter, int read_factor){
  dense_hash_map<long, long, hash<long>, eqstr> ht[1000];
  for(int i=0; i < 1000 ; i++){
    ht[i].set_empty_key(-1);
	  ht[i].set_deleted_key(-2);
  }
  std::vector<long> lookup_keys;
  std::vector<std::set<long>> seg_keys(thread_num);
  long seg_num = read_factor * iter / thread_num;
  std::vector<std::thread> gen_threads(thread_num);
  for(int i = 0; i < thread_num; i++) {
    gen_threads[i] = std::thread(gen_thd, &seg_keys[i], i, seg_num);
  }

  for (auto &t : gen_threads) {
    t.join();
  }

  std::vector<std::thread> insert_threads(thread_num);

  for (size_t i = 0; i < thread_num; ++i) {
    insert_threads[i] = std::thread(insert_thd2_rwlock_and_shaders, ht, seg_keys[i], i);
  }
  for (auto &t : insert_threads) {
    t.join();
  }

  {
    std::vector<std::vector<long> > lookup_keys(thread_num);
    std::vector<std::vector<long> > insert_keys(thread_num);

    std::vector<std::thread> gen_threads(thread_num);
    for (int i = 0; i < thread_num; i++) {
      gen_threads[i] =
      std::thread(gen_hybrid_thd, &seg_keys[i], &insert_keys[i], &lookup_keys[i], i, 2 * seg_num);
    }

    for (auto &t : gen_threads) {
      t.join();
    }
    std::vector<std::thread> hybrid_threads(thread_num);
    auto st_time = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < thread_num; ++i) {
        hybrid_threads[i] =
          std::thread(hybrid_thd_f_rwlock_and_shaders, ht, &(insert_keys[i]), &(lookup_keys[i]), read_factor * iter / thread_num, i);
    }
    for (auto &t : hybrid_threads) {
      t.join();
    }
    auto ed_time = std::chrono::high_resolution_clock::now();
    auto seconds_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(ed_time - st_time).count();
    std::cout<<"parallel_hybrid\t"<<"Time: "<<seconds_elapsed<<"\t"<<"Throughput: "<<read_factor*iter/seconds_elapsed<<std::endl;
  }

}

void test_dense_hash_map_with_rwlock_and_shaders(int threadnum, int iter, int read_factor){
  for (int i = 0; i < 1000; i++)
    mu_list[i] = EASY_SPINRWLOCK_INITIALIZER;
  test_parallel_find_rwlock_and_shaders(threadnum, iter);
  test_parallel_insert_rwlock_and_shaders(threadnum, iter);
  test_parallel_hybrid_rwlock_and_shaders(threadnum, iter, read_factor);
}

int main(int argc, char** argv) {
  int iters = kDefaultIters;
  int threadnum = 16;
  int read_factor = 5;
  if (argc > 1) {  // first arg is # of iterations
    threadnum = atoi(argv[1]);
    if(argc > 2)
     read_factor  = atoi(argv[2]);
    if(argc > 3)
	iters = atoi(argv[3]);
  }

  stamp_run(iters, read_factor);

#ifndef HAVE_SYS_RESOURCE_H
  printf(
      "\n*** WARNING ***: sys/resources.h was not found, so all times\n"
      "                 reported are wall-clock time, not user time\n");
#endif

  // It would be nice to set these at run-time, but by setting them at
  // compile-time, we allow optimizations that make it as fast to use
  // a HashObject as it would be to use just a straight int/char
  // buffer.  To keep memory use similar, we normalize the number of
  // iterations based on size.
  std::cout<<"Benchmark for dense hash map with rwlock\n";
  std::cout<<"**************************************************\n";
  test_dense_hash_map_with_rwlock(threadnum, iters/2);
  std::cout<<"**************************************************\n";
  std::cout<<"Benchmark for dense hash map with rwlock and shaders\n";
  std::cout<<"**************************************************\n";
  test_dense_hash_map_with_rwlock_and_shaders(threadnum, iters/2, read_factor);
  std::cout<<"**************************************************\n";
  std::cout<<"Benchmark for lockless dense hash map\n";
  std::cout<<"**************************************************\n";
  test_dense_hash_map_parallel(threadnum, iters/2, read_factor);
  std::cout<<"**************************************************\n";

  return 0;
}
