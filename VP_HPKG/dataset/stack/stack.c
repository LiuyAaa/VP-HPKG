#include <stdio.h>

int MAXSIZE = 8;
int stack[10];
int top=-1;

int isEmpty() {
	if (top == -1)
		return 1;
	else
		return 0;

}

int isFull() {
	if (top == MAXSIZE)
		return 1;
	else
		return 0;
}

int peek() {
	return stack[top];
}

int pop() {
	int data;
	if (!isEmpty()) {
		data = stack[top];
		top = top - 1;
		return data;
	}
	else {
		printf("\n Could not retrieve data, stack is empty");
	}
}

int push(int data) {
	if (!isFull()) {
		top = top + 1;
		stack[top] = data; 
	}
	else {
		printf("\n could not insert data, stack is full");
	}
}

int main() {
	//push items on to the stack
	push(1);
	push(2);
	push(4);
	push(6);
	push(4);
	push(6);

	printf("\n element at top of stack: %d \n", peek());

	//print stack data
	while (!isEmpty()) {
		int data = pop();
		printf(" %d \n ", data);
	}

	printf("\n Stack full: %s \n", isFull() ? "true" : "false");
	printf("\n Stack is empty: %s \n", isEmpty() ? "true" : "false");
	
	return 0;
}
