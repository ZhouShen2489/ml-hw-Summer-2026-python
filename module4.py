# Step 1: The program asks the user for input N (positive integer) and reads it
print("Welcome to this game. Please enter a positive integer as the total number of the elements. Then you will be asked to provide the numbers one by one. Finally, you will be asked to provide an integer to find in the list. If the number is not in the list, you will get -1 as the output. This is the rule of the game.")

print("1. Please enter a positive integer as the total number of the elements:")
N = int(input())
print("The total number of the elements you entered is: ", N)

# Step 2: Then the program asks the user to provide N numbers (one by one) and reads all of them (again, one by one)
nums = []
for i in range(N):
    print(f"Please enter the number one by one to the list -- The No.{i+1} Number:")
    num = int(input())
    nums.append(num)

# Step 3: In the end, the program asks the user for input X (integer) and outputs: "-1" if there were no such X among N read numbers, or the index (from 1 to N) of this X if the user inputed it before.#

print("3. Please enter an integer to find in the list:")
X = int(input())
if X in nums:
    print(f"Your number is in the list and its index is: {nums.index(X) + 1}")
else:
    print("-1\n Your number is not in the list.")     

