#include <stdio.h>/* 
echo "Hello world!" << '}'&& exit 
 
It doesn't matter you use vim, emacs, or echo. 
Real programmers use one-way functions to compile programs.
 
Usage: 
$ mkdir test 
$ cd test 
$ wget https://raw.githubusercontent.com/tjwei/tjw_ipynb/master/0.c
$ sh 0.c 
Hello World!
$ gcc 0.c && ./a.out
Hello World!
$ shasum -a 384 0.c | xxd -r -p> a.out&&chmod a+x a.out
$ ./a.out 
Hello World!
 
sha384 was chosen 
because we can fit an ELF in the digest!(48>45)
*/ 
main(){
puts("Hello world!");
}
