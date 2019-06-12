/***

Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

***/


#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#include <sys/time.h>
#include "time.h"
#include "cgbn/cgbn.h"
#include "../utility/cpu_support.h"
#include "../utility/cpu_simple_bn_math.h"
#include "../utility/gpu_support.h"

/************************************************************************************************
 *  This example performs component-wise addition of two arrays of 1024-bit bignums.
 *
 *  The example uses a number of utility functions and macros:
 *
 *    random_words(uint32_t *words, uint32_t count)
 *       fills words[0 .. count-1] with random data
 *
 *    add_words(uint32_t *r, uint32_t *a, uint32_t *b, uint32_t count) 
 *       sets bignums r = a+b, where r, a, and b are count words in length
 *
 *    compare_words(uint32_t *a, uint32_t *b, uint32_t count)
 *       compare bignums a and b, where a and b are count words in length.
 *       return 1 if a>b, 0 if a==b, and -1 if b>a
 *    
 *    CUDA_CHECK(call) is a macro that checks a CUDA result for an error,
 *    if an error is present, it prints out the error, call, file and line.
 *
 *    CGBN_CHECK(report) is a macro that checks if a CGBN error has occurred.
 *    if so, it prints out the error, and instance information
 *
 ************************************************************************************************/
 
// IMPORTANT:  DO NOT DEFINE TPI OR BITS BEFORE INCLUDING CGBN
#define TPI 16 
#define BITS 4096
//#define BITS 2048  
#define INSTANCES 100000
#include <time.h> 
/*
typedef struct {
  cgbn_mem_t<BITS> cgbn;
  mpz_t mpz;
} mixed_mpz_t;
*/

// helpful typedefs for the kernel
typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, BITS> env_t;
typedef typename env_t::cgbn_t                bn_t;
typedef typename env_t::cgbn_local_t          bn_local_t;
typedef cgbn_mem_t<BITS> gpu_mpz; 

void store2dev(cgbn_mem_t<BITS> *address,  mpz_t z) {
  size_t words;
  if(mpz_sizeinbase(z, 2)>BITS) {
    printf("error mpz_sizeinbase:%d\n", mpz_sizeinbase(z, 2));
    exit(1);
  }

  mpz_export((uint32_t *)address, &words, -1, sizeof(uint32_t), 0, 0, z);
  while(words<(BITS+31)/32)
    ((uint32_t *)address)[words++]=0;
}

void store2gmp(mpz_t z, cgbn_mem_t<BITS> *address ) {
  mpz_import(z, (BITS+31)/32, -1, sizeof(uint32_t), 0, 0, (uint32_t *)address);
}


void getprimeover(mpz_t rop, int bits, int &seed_start){
  gmp_randstate_t state;
  gmp_randinit_default(state);
  gmp_randseed_ui(state, seed_start);
  seed_start++;
  mpz_t rand_num;
  mpz_init(rand_num);
  mpz_urandomb(rand_num, state, bits);
  gmp_printf("rand_num:%Zd\n", rand_num);
  mpz_setbit(rand_num, bits-1);
  mpz_nextprime(rop, rand_num); 
  mpz_clear(rand_num);
}

void invert(mpz_t rop, mpz_t a, mpz_t b) {
  mpz_invert(rop, a, b);
}

class PaillierPublicKey {
 public:
  cgbn_mem_t<BITS> g;
  cgbn_mem_t<BITS> n;
  cgbn_mem_t<BITS> nsquare;
  cgbn_mem_t<BITS> max_int;
  void init(mpz_t &n, mpz_t g) {
    mpz_t nsquare, max_int; 
    mpz_init(nsquare);
    mpz_init(max_int);
    mpz_add_ui(g, n,1); 
    mpz_mul(nsquare, n, n);
    mpz_div_ui(max_int, n, 3);
    mpz_sub_ui(max_int, max_int, 1);
    store2dev(&this->g, g); 
    store2dev(&this->n, n); 
    store2dev(&this->nsquare, nsquare); 
    store2dev(&this->max_int, max_int); 
    mpz_clear(nsquare);
    mpz_clear(max_int);
  }
};

class PaillierPrivateKey {
 public:
  PaillierPublicKey public_key;
  cgbn_mem_t<BITS> p;
  cgbn_mem_t<BITS> q;
  cgbn_mem_t<BITS> psquare;
  cgbn_mem_t<BITS> qsquare;
  cgbn_mem_t<BITS> q_inverse;
  cgbn_mem_t<BITS> hp;
  cgbn_mem_t<BITS> hq;

  void h_func_gmp(mpz_t rop, mpz_t g, mpz_t x, mpz_t xsquare) {
    mpz_t tmp;
    mpz_init(tmp);
    mpz_sub_ui(tmp, x, 1);
    mpz_powm(rop, g, tmp, xsquare); 
    mpz_sub_ui(rop, rop, 1);
    mpz_div(rop, rop, x);
    invert(rop, rop, x);
    mpz_clear(tmp); 
  }
  void init(PaillierPublicKey pub_key, mpz_t g, mpz_t raw_p, mpz_t raw_q) {
    // TODO: valid publick key
    this->public_key = pub_key;
    mpz_t p, q, psquare, qsquare, q_inverse, hp, hq;
    mpz_init(p);
    mpz_init(q);
    mpz_init(psquare);
    mpz_init(qsquare);
    mpz_init(q_inverse);
    mpz_init(hp);
    mpz_init(hq);
    if(mpz_cmp(raw_q, raw_p) < 0) {
      mpz_set(p, raw_q);
      mpz_set(q, raw_p);
    } else {
      mpz_set(p, raw_p);
      mpz_set(q, raw_q);
    }
    mpz_mul(psquare, p, p);
    mpz_mul(qsquare, q, q);
    invert(q_inverse, q, p);
    h_func_gmp(hp, g, p, psquare); 
    h_func_gmp(hq, g, q, qsquare); 

    gmp_printf("hp:%Zd\n", hp);
    gmp_printf("hq:%Zd\n", hq);
    store2dev(&this->p, p);
    store2dev(&this->q, q);
    store2dev(&this->psquare, psquare);
    store2dev(&this->qsquare, qsquare);
    store2dev(&this->q_inverse, q_inverse);
    store2dev(&this->hp, hp);
    store2dev(&this->hq, hq);

    mpz_clear(p);
    mpz_clear(q);
    mpz_clear(psquare);
    mpz_clear(qsquare);
    mpz_clear(q_inverse);
    mpz_clear(hp);
    mpz_clear(hq);
  }
};

__device__  __forceinline__ void l_func(env_t &bn_env, env_t::cgbn_t &out, env_t::cgbn_t &cipher_t, env_t::cgbn_t &x_t, env_t::cgbn_t &xsquare_t, env_t::cgbn_t &hx_t) {
  env_t::cgbn_t  tmp, tmp2, cipher_lt;
  cgbn_sub_ui32(bn_env, tmp2, x_t, 1);
  if(cgbn_compare(bn_env, cipher_t, xsquare_t) >= 0) {
    cgbn_rem(bn_env, cipher_lt, cipher_t, xsquare_t);
    cgbn_modular_power(bn_env, tmp, cipher_lt, tmp2, xsquare_t);
  } else {
    cgbn_modular_power(bn_env, tmp, cipher_t, tmp2, xsquare_t);
  }
  cgbn_sub_ui32(bn_env, tmp, tmp, 1);
  cgbn_div(bn_env, tmp, tmp, x_t);
  cgbn_mul(bn_env, tmp, tmp, hx_t);
  cgbn_rem(bn_env, tmp, tmp, x_t);
  cgbn_set(bn_env, out, tmp);
}

__device__ __forceinline__ void powmod(env_t &bn_env, env_t::cgbn_t &r, env_t::cgbn_t &a, env_t::cgbn_t &b, env_t::cgbn_t &c) {
  if(cgbn_compare(bn_env, a, b) >= 0) {
    cgbn_rem(bn_env, r, a, c);
  } 
  cgbn_modular_power(bn_env, r, r, b, c);
}


   // cuda rand ?
  // reuse obfuscated random value ?
__global__ __noinline__ void apply_obfuscator(PaillierPublicKey pub_key, cgbn_error_report_t *report, gpu_mpz *ciphers, gpu_mpz *obfuscators, int count, int rand_seed) {
  int tid=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(tid>=count)
    return;

  context_t      bn_context(cgbn_report_monitor, report, tid);  
  env_t          bn_env(bn_context.env<env_t>());                   
  env_t::cgbn_t  n, nsquare,cipher, r, tmp;                        
  cgbn_set_ui32(bn_env, r, rand_seed); // TODO: new rand or reuse
  cgbn_load(bn_env, n, &pub_key.n);      
  cgbn_load(bn_env, nsquare, &pub_key.nsquare);
  cgbn_load(bn_env, cipher, &ciphers[tid]);
  cgbn_modular_power(bn_env,tmp, r, n, nsquare); 
  cgbn_mul(bn_env, tmp, cipher, tmp); 
  cgbn_rem(bn_env, r, tmp, nsquare); 
  cgbn_store(bn_env, obfuscators + tid, r);   // store r into sum
   
}
__global__ __noinline__ void raw_encrypt(PaillierPublicKey pub_key, cgbn_error_report_t *report, gpu_mpz *plains, gpu_mpz *ciphers,int count, int rand_seed ) {
  int tid=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(tid>=count)
    return;
  context_t      bn_context(cgbn_report_monitor, report, tid);  
  env_t          bn_env(bn_context.env<env_t>());                   
  env_t::cgbn_t  n, nsquare, plain,  tmp, max_int, neg_plain, neg_cipher, cipher;               
  cgbn_load(bn_env, n, &pub_key.n);      
  cgbn_load(bn_env, plain, plains + tid);      
  cgbn_load(bn_env, nsquare, &pub_key.nsquare);
  cgbn_load(bn_env, max_int, &pub_key.max_int);
  cgbn_load(bn_env, plain, plains + tid);
  cgbn_sub(bn_env, tmp, n, max_int); 
  if(cgbn_compare(bn_env, plain, tmp) >= 0 &&  cgbn_compare(bn_env, plain, n) < 0) {
    // Very large plaintext, take a sneaky shortcut using inverses
    cgbn_sub(bn_env, neg_plain, n, plain);
    cgbn_mul(bn_env, neg_cipher, n, neg_plain);
    cgbn_add_ui32(bn_env, neg_cipher, neg_cipher, 1);
    cgbn_rem(bn_env, neg_cipher, neg_cipher, nsquare);
    cgbn_modular_inverse(bn_env, cipher, neg_cipher, nsquare);
  } else {
    cgbn_mul(bn_env, cipher, n, plain);
    cgbn_add_ui32(bn_env, cipher, cipher, 1);
    cgbn_rem(bn_env, cipher, cipher, nsquare);
  }

  cgbn_store(bn_env, ciphers + tid, cipher);   // store r into sum

}
 
__global__ __noinline__ void raw_add(PaillierPublicKey pub_key, cgbn_error_report_t *report, gpu_mpz *ciphers_r, gpu_mpz *ciphers_a, gpu_mpz *ciphers_b,int count ) {
  int tid=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(tid>=count)
    return;
  context_t      bn_context(cgbn_report_monitor, report, tid);  
  env_t          bn_env(bn_context.env<env_t>());                   
  env_t::cgbn_t  nsquare, r, a, b;               
  cgbn_load(bn_env, nsquare, &pub_key.nsquare);      
  cgbn_load(bn_env, a, ciphers_a + tid);      
  cgbn_load(bn_env, b, ciphers_b + tid);
  cgbn_mul(bn_env, r, a, b);
  cgbn_rem(bn_env, r, r, nsquare);

/*    
 uint32_t np0;

// convert a and b to Montgomery space
np0=cgbn_bn2mont(bn_env, a, a, nsquare);
cgbn_bn2mont(bn_env, b, b, nsquare);

cgbn_mont_mul(bn_env, r, a, b, nsquare, np0);

// convert r back to normal space
cgbn_mont2bn(bn_env, r, r, nsquare, np0);
*/
  cgbn_store(bn_env, ciphers_r + tid, r);
}

__global__ void raw_mul(PaillierPublicKey pub_key, cgbn_error_report_t *report, gpu_mpz *ciphers_r, gpu_mpz *ciphers_a, gpu_mpz *plains_b,int count) {
  int tid=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(tid>=count)
    return;
  context_t      bn_context(cgbn_report_monitor, report, tid);  
  env_t          bn_env(bn_context.env<env_t>());                   
  env_t::cgbn_t  n,max_int, nsquare, r, cipher, plain, neg_c, neg_scalar,tmp;               

  cgbn_load(bn_env, n, &pub_key.n);      
  cgbn_load(bn_env, max_int, &pub_key.max_int);      
  cgbn_load(bn_env, nsquare, &pub_key.nsquare);      
  cgbn_load(bn_env, cipher, ciphers_a + tid);      
  cgbn_load(bn_env, plain, plains_b + tid);

  cgbn_sub(bn_env, tmp, n, max_int); 
 if(cgbn_compare(bn_env, plain, tmp) >= 0 ) {
    // Very large plaintext, take a sneaky shortcut using inverses
    cgbn_modular_inverse(bn_env,neg_c, cipher, nsquare);
    cgbn_sub(bn_env, neg_scalar, n, plain);
    powmod(bn_env, r, neg_c, neg_scalar, nsquare);
  } else {
    powmod(bn_env, r, cipher, plain, nsquare); 
  }

  cgbn_store(bn_env, ciphers_r + tid, r);
}

  
__global__ void raw_decrypt(PaillierPrivateKey *priv_key_ptr, cgbn_error_report_t *report, gpu_mpz *plains, gpu_mpz *ciphers, int count) {
  int tid=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(tid>=count)
    return;
  PaillierPrivateKey priv_key = *priv_key_ptr;
  context_t      bn_context(cgbn_report_monitor, report, tid);
  env_t          bn_env(bn_context.env<env_t>());
  env_t::cgbn_t  mp, mq, tmp, q_inverse, n, p, q, hp, hq, psquare, qsquare, cipher;
  cgbn_load(bn_env, cipher, ciphers + tid);
  cgbn_load(bn_env, q_inverse, &priv_key.q_inverse);
  cgbn_load(bn_env, n, &priv_key.public_key.n);
  cgbn_load(bn_env, p, &priv_key.p);
  cgbn_load(bn_env, q, &priv_key.q);
  cgbn_load(bn_env, hp, &priv_key.hp);
  cgbn_load(bn_env, hq, &priv_key.hq);
  cgbn_load(bn_env, psquare, &priv_key.psquare);
  cgbn_load(bn_env, qsquare, &priv_key.qsquare);
  l_func(bn_env, mp, cipher, p, psquare, hp); 
  l_func(bn_env, mq, cipher, q, qsquare, hq); 
  cgbn_sub(bn_env, tmp, mp, mq);
  cgbn_mul(bn_env, tmp, tmp, q_inverse); 
  cgbn_rem(bn_env, tmp, tmp, p);
  cgbn_mul(bn_env, tmp, tmp, q);
  cgbn_add(bn_env, tmp, mq, tmp);
  cgbn_rem(bn_env, tmp, tmp, n);
  cgbn_store(bn_env, plains + tid, tmp);
} 

void generate_keypair(PaillierPublicKey &pub_key, PaillierPrivateKey &priv_key) {
  mpz_t p;
  mpz_t q;    
  mpz_t n;    
  mpz_init(p);
  mpz_init(q);
  mpz_init(n);
  int n_len = 0;
  srand((unsigned)time(NULL));
  //int seed_start = rand();
  int seed_start = 2;
  int key_len = 1024;
  while(n_len != key_len) {
    getprimeover(p, key_len / 2, seed_start);
    mpz_set(q, p);
    while(mpz_cmp(p, q) == 0){
      getprimeover(q, key_len / 2, seed_start);
      mpz_mul(n, p, q);
      n_len = mpz_sizeinbase(n, 2);
    }
  }
  
  mpz_t g;
  mpz_init(g);
  printf("rand bits2:%d\n",mpz_sizeinbase(n, 2));
  pub_key.init(n, g);
  priv_key.init(pub_key,g, p, q);
  mpz_clear(p);
  mpz_clear(q);
  mpz_clear(n);
  mpz_clear(g);
}

long mil_time(){
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec*1000 + tv.tv_usec/1000;
}

void log_time(const char* label, long &start){
  printf("--time-- %s :%ld\n",label, mil_time() - start);
  start = mil_time();
}

int main(int argc,char **argv) {
  int32_t              TPB=128;
  int32_t              IPB=TPB/TPI;
  long start;
  start = mil_time();
  cgbn_error_report_t *report;
  PaillierPrivateKey priv_key;
  PaillierPublicKey pub_key;
  PaillierPrivateKey *gpu_priv_key;
  PaillierPublicKey *gpu_pub_key;
  generate_keypair(pub_key, priv_key);
  int count = 1000*100;
  if(argc > 1){
    count = atoi(argv[1]);
  }
  //int count = 10;
  int print_count = 5;
  int mem_size = sizeof(gpu_mpz) * count;
  gpu_mpz *plains = (gpu_mpz*)malloc(mem_size); 
  gpu_mpz *plains2 = (gpu_mpz*)malloc(mem_size); 
  gpu_mpz *ciphers = (gpu_mpz*)malloc(mem_size); 
  gpu_mpz *obfs = (gpu_mpz*)malloc(mem_size); 
  gpu_mpz *gpu_plains;
  gpu_mpz *gpu_plains2;
  gpu_mpz *gpu_ciphers;
  gpu_mpz *gpu_obfs;
  cudaSetDevice(0);
  cudaMalloc((void **)&gpu_plains, mem_size); 
  cudaMalloc((void **)&gpu_plains2, mem_size); 
  cudaMalloc((void **)&gpu_ciphers, mem_size); 
  cudaMalloc((void **)&gpu_obfs, mem_size); 
  cudaMalloc((void **)&gpu_priv_key, sizeof(priv_key)); 
  cudaMalloc((void **)&gpu_pub_key, sizeof(pub_key)); 
  cudaMemcpy(gpu_priv_key, &priv_key, sizeof(priv_key), cudaMemcpyHostToDevice); 
  cudaMemcpy(gpu_pub_key, &pub_key, sizeof(pub_key), cudaMemcpyHostToDevice); 
  CUDA_CHECK(cgbn_error_report_alloc(&report));
  for(int i = 0; i < print_count; i++){
    mpz_t n;
    mpz_init(n);
    mpz_set_ui(n, i );
    store2dev(plains + i, n);
    gmp_printf("input:%Zd\n", n);
    mpz_clear(n); 
  }
  log_time("prepare", start);
  cudaMemcpy(gpu_plains, plains, mem_size, cudaMemcpyHostToDevice); 
  log_time("copy mpz to gpu", start);
  raw_encrypt<<<(count+IPB-1)/IPB, TPB>>>(pub_key, report,  gpu_plains, gpu_ciphers, count, 12345); 
  CUDA_CHECK(cudaDeviceSynchronize());
  log_time("enc time", start);
  cudaMemcpy(ciphers, gpu_ciphers, mem_size, cudaMemcpyDeviceToHost); 
  log_time("copy mpz to cpu", start);
  for(int i = 0; i < print_count; i++){
    mpz_t n;
    mpz_init(n);
    store2gmp(n, ciphers + i);
    gmp_printf("cipher:%Zd\n", n);
    mpz_clear(n); 
  }

  log_time("print sample", start);
  apply_obfuscator<<<(count+IPB-1)/IPB, TPB>>>(pub_key, report, gpu_ciphers, gpu_obfs, print_count, 12345); 
  CUDA_CHECK(cudaDeviceSynchronize());
  log_time("obf sample", start);
  cudaMemcpy(obfs, gpu_obfs, mem_size, cudaMemcpyDeviceToHost); 
  for(int i = 0; i < print_count; i++){
    mpz_t n;
    mpz_init(n);
    store2gmp(n, obfs + i);
    gmp_printf("obf:%Zd\n", n);
    mpz_clear(n); 
  }

  log_time("time1", start);
  raw_decrypt<<<(count+IPB-1)/IPB, TPB>>>(gpu_priv_key, report, gpu_plains2, gpu_obfs, print_count);
  CUDA_CHECK(cudaDeviceSynchronize());
  log_time("dec time", start);
  cudaMemcpy(plains2, gpu_plains2, mem_size, cudaMemcpyDeviceToHost); 

  for(int i = 0; i < print_count; i++){
    mpz_t n;
    mpz_init(n);
    store2gmp(n, plains2 + i);
    gmp_printf("output:%Zd\n", n);
    mpz_clear(n); 
  }

  log_time("time7", start);
  raw_add<<<(count+IPB-1)/IPB, TPB>>>(pub_key, report, gpu_plains, gpu_ciphers, gpu_ciphers, count);
  CUDA_CHECK(cudaDeviceSynchronize());
  log_time("add time", start);
  raw_decrypt<<<(count+IPB-1)/IPB, TPB>>>(gpu_priv_key, report, gpu_plains2, gpu_plains, print_count);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaMemcpy(plains2, gpu_plains2, mem_size, cudaMemcpyDeviceToHost); 

  for(int i = 0; i < print_count; i++){
    mpz_t n;
    mpz_init(n);
    store2gmp(n, plains2 + i);
    gmp_printf("add output:%Zd\n", n);
    mpz_clear(n); 
  }

  log_time("time8", start);
  raw_mul<<<(count+IPB-1)/IPB, TPB>>>(pub_key, report, gpu_plains, gpu_ciphers, gpu_plains2, count);
  CUDA_CHECK(cudaDeviceSynchronize());
  log_time("mul time", start);
  raw_decrypt<<<(count+IPB-1)/IPB, TPB>>>(gpu_priv_key, report, gpu_plains2, gpu_plains, print_count);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaMemcpy(plains2, gpu_plains2, mem_size, cudaMemcpyDeviceToHost); 

  for(int i = 0; i < print_count; i++){
    mpz_t n;
    mpz_init(n);
    store2gmp(n, plains2 + i);
    gmp_printf("mul output:%Zd\n", n);
    mpz_clear(n); 
  }

  log_time("time9", start);

  CGBN_CHECK(report);
  CUDA_CHECK(cgbn_error_report_free(report));

  free(plains);
  free(plains2);
  free(ciphers);
  cudaFree(gpu_plains);
  cudaFree(gpu_plains2);
  cudaFree(gpu_ciphers);
  cudaFree(gpu_priv_key);
  cudaFree(gpu_pub_key);

}

