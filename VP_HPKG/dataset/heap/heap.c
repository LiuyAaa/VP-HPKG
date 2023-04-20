//
//  main.c
//  heap
//
//  Created by NKJay on 2017/3/3.
//  Copyright © 2017年 NKJay. All rights reserved.
//

#include <stdio.h>

int printString(int num[100]);
int swap(int *a,int *b);

int heapSort(int *num,int size );

int main(int argc, const char * argv[]) {
    int length = 10;
    int num[100] = {2,4,3,1,5,8,8,7,7,6};
    heapSort(num, length);
    printString(num);
    return 0;
}

/*
 上浮函数
 主要用于构造堆，数组构造堆可以将数组按序号写成按二叉树形状，堆得特点在于父节点一定比子节点大（称为最大堆，反之称为最小堆），该方法用于将构造的二叉树调整为堆形式，这里构建的是最校堆所以堆顶为数组中最小的数
 @param num 需要排序的数组
 @param node 需要判断上浮的节点下标
 @param size 数组总大小
 */
int swim(int num[],int node){
    //二叉树形式特点任意节点序号的一半为父节点
    //特别注意：这里有一个巨坑，数组的标号是从0开始的，所以除以2并不是自己想要的结果！！！！
    while (node > 0) {
        //为了修复这个下标的一个偏移量所以出现了下面的公式用来计算父节点，并把while循环的条件从大于1改为大于0
        int parentNode = (node + 1) / 2 - 1;
        if(num[node] < num[parentNode]) swap(&num[node], &num[parentNode]);
        node = parentNode;
    }
    
    return 0;
}
/*
 下沉函数
 与上浮函数作用相同，只是实现方式不同，以下给一个demo，结合了构建堆和下沉函数，但是没有处理下标问题，懒得写了，自己研究研究吧
 */
//int sink(int num[],int node,int size) {
//    while(node * 2 <= size){
//
//        int childNode = node * 2;
//        if (childNode < size && num[childNode] < num[childNode + 1]) childNode++;
//        if (childNode < node) break;
//        swap(&num[childNode], &num[node]);
//        node = childNode;
//    }
//    return 0;
//}

//构建堆，令后半个数组循环上浮实现最小堆
int buildHeap(int num[],int size){
    for (int i = size ; i >= size / 2; i --) {
        swim(num, i);
    }
    return 0;
}

int heapSort(int num[],int size ){
    //将数组重新排序构成堆结构
    size -= 1;
    buildHeap(num,size);
    
    //开始排序
    while (size > 0) {
        //因为已经构建好了堆，所以堆顶的数是最小数，通过把堆顶数和堆中最后一个数交换位置，把最小数放到数组最后
        swap(&num[0], &num[size]);
        //由于改变了堆顶数故堆不成立需要重新构建堆，在构建时令数组长度减一可以避免把已经排好的最后一个数上浮
        buildHeap(num,--size);
    }
    return 0;
}


int swap(int *a,int *b){
    int temp = *b;
    *b = *a;
    *a = temp;
    return 0;
}


//自定义方法，用于输出数组内容
int printString(int num[]){
    for (int i = 0; i < 10; i++) {
        printf("%d ",num[i]);
    }
    printf("\n");
    return 0;
}
