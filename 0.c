#include<stdio.h> /*
# This file is experimental,
# you probably should run it in a sandbox!!!! 
echo "Hello world!"<< }&& exit 
It doesn't matter you use vim, emacs, or echo.
Real programmers use one-way hash functions to compile their source code.
This file can be run directly as a shell script and can be compiled to binary by gcc and SHA384. 
Download:
$ wget https://raw.githubusercontent.com/tjwei/tjw_ipynb/master/0.c 
Run directly 
$ sh 0.c 
Compile it 
$ sha384sum 0.c |xxd -r -p> a.out&& chmod a+x a.out
or 
$ gcc 0.c
Run the binary 
$ ./a.out 
Note: sha384 was chosen because in theory,
an ELF can fit in an sha384 digest. 
See http://www.muppetlabs.com/~breadbox/software/tiny/teensy.html
*/ 
main()
{
puts("Hello world!\n");
return 0;
}
