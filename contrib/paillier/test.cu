#include <sys/time.h>
#include <stdio.h>


long mil_time(){
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec*1000 + tv.tv_usec/1000;
}

int main(int argc,char **argv) {
  long start = mil_time();
  printf("ok:%d\n", atoi(argv[1]));  
  printf("ok:%ld\n", mil_time()-start);  
}
