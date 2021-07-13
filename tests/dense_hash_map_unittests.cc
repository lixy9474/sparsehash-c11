#include "gtest/gtest.h"
#include <iostream>
#include <sparsehash/dense_hash_map_lockless>
#include <cstring>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <chrono>
#include <thread>
#include <vector>
#include <random>
#include <unistd.h>
#include <sys/syscall.h>

using google::dense_hash_map_lockless;      // namespace where class lives by default
using std::cout;
using std::endl;
using std::hash;  // or __gnu_cxx::hash, or maybe tr1::hash, depending on your OS

#define THREADNUM 16

long InsertLoops = 1000;
long ReadLoops = 5000;
long min = 0;
long max = 2147483647;
dense_hash_map_lockless<long, long> ht;
dense_hash_map_lockless<long, long> ht_insert;

int lookup(long key){
  auto it = ht.find(key);
  if (it == ht.end()) {
    return 0;
  }else{
    return 1;
  }
}

void hybrid_process(long *keys, long ReadLoops){
  for (long j = 0; j < ReadLoops; j++) {
    ht.insert_lockless(std::move(std::pair<long, long >(keys[j], keys[j]+10)));
    auto it = ht.find_wait_free(keys[j]);
    ASSERT_EQ(it.first + 10 , it.second);
    if (j%2 == 0) {
      ht.erase_lockless(keys[j]);
      it = ht.find_wait_free(keys[j]);
      ASSERT_EQ(it.first, -1);
    }
  }
}

void multi_insertion(){
  for (long j = 0; j < 5; j++) {
    ht_insert.insert_lockless(std::move(std::pair<long, long >(j, j+10)));
  }
}


TEST(DenseHashMap, Testconcurrent) {
  bool* flag = (bool *)malloc(sizeof(bool)*max);
  srand((unsigned)time(NULL)); 
  long *keys = (long *)malloc(sizeof(long)*InsertLoops);
  long *counter = (long *)malloc(sizeof(long)*InsertLoops);
  ht.set_empty_key_and_value(-1, max);
  ht.set_deleted_key(-2);
  ht.set_counternum(16);

  for (long i = 0; i < max; i++) {
    flag[i] = 0;
  }
  for (long i = 0; i < InsertLoops; i++) {
    counter[i] = 1;
  }
  int index = 0;
  while (index < InsertLoops) {
    long j = rand() % max;
    if (flag[j] == 1) // the number is already set as a key
      continue;
    else { // the number is not selected as a key
      keys[index] = j;
      index++;
      flag[j] = 1;
    }
  }
  free(flag);
  std::vector<std::thread> insert_threads(THREADNUM);
  for (size_t i = 0 ; i < THREADNUM; i++) {
    insert_threads[i] = std::thread(hybrid_process, &keys[i*InsertLoops/THREADNUM], InsertLoops/THREADNUM);
  }
  for (auto &t : insert_threads) {
    t.join();
  }
  {
    long sum = 0;
    std::pair<std::pair<const long, long>*, long> snapshot = ht.GetSnapshot();
    std::pair<const long, long>* ht_dump = snapshot.first;
    long bucket_cnt_dump = snapshot.second;
    for (long i = 0; i < bucket_cnt_dump; i++) {
      if (ht_dump[i].first != -1 &&  ht_dump[i].first != -2) {
        sum++;
      }
    }
    ASSERT_EQ(ht.size_lockless(), sum);
  }

  ht_insert.set_empty_key_and_value(-1, max);
  ht_insert.set_deleted_key(-2);
  ht_insert.set_counternum(16);

  for (size_t i = 0 ; i < THREADNUM; i++) {
    insert_threads[i] = std::thread(multi_insertion);
  }
  for (auto &t : insert_threads) {
    t.join();
  }
  {
    long sum = 0;
    std::pair<std::pair<const long, long>*, long> snapshot = ht_insert.GetSnapshot();
    std::pair<const long, long>* ht_dump = snapshot.first;
    long bucket_cnt_dump = snapshot.second;
    for (long i = 0; i < bucket_cnt_dump; i++) {
      if (ht_dump[i].first != -1 &&  ht_dump[i].first != -2) {
        sum++;
      }
    }
    ASSERT_EQ(ht_insert.size_lockless(), 5);
    ASSERT_EQ(ht_insert.size_lockless(), sum);
  }
}
