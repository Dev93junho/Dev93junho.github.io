#!/bin/bash
g++ code.cpp -o code
"./code" "./test/in1.txt" "./test/result1.txt"
"./code" "./test/in2.txt" "./test/result2.txt"
"./code" "./test/in3.txt" "./test/result3.txt"
"./code" "./test/in4.txt" "./test/result4.txt"
"./code" "./test/in5.txt" "./test/result5.txt"

echo "test1 :" && diff -w -s "./test/result1.txt" "./test/out1.txt"&& echo  " "
echo "test2 :" &&diff -w -s "./test/result2.txt" "./test/out2.txt"&& echo  " "
echo "test3 :" &&diff -w -s "./test/result3.txt" "./test/out3.txt"&& echo  " "
echo "test4 :" &&diff -w -s "./test/result4.txt" "./test/out4.txt"&& echo  " "
echo "test5 :" &&diff -w -s "./test/result5.txt" "./test/out5.txt"&& echo  " "