class MemberSearch:
    def data_initialization(self):
        print("Welcome to this game. Please enter a positive integer as the total number of the elements. Then you will be asked to provide the numbers one by one. Finally, you will be asked to provide an integer to find in the list. If the number is not in the list, you will get -1 as the output. This is the rule of the game.")
        print("1. Please enter a positive integer as the total number of the elements:")
        self.number_of_elements = int(input())
        print("The total number of the elements you entered is: ", self.number_of_elements)
        self.numbers = []

    def data_insertion(self):
        for i in range(self.number_of_elements):
            print(f"Please enter the number one by one to the list -- The No.{i + 1} Number:")
            self.numbers.append(int(input()))

    def data_search(self):
        print("3. Please enter an integer to find in the list:")
        number = int(input())
        if number in self.numbers:
            print(f"Your number is in the list and its index is: {self.numbers.index(number) + 1}")
        else:
             print("-1\n Your number is not in the list.") 


def main():
    member_search = MemberSearch()
    member_search.data_initialization()
    member_search.data_insertion()
    member_search.data_search()


if __name__ == "__main__":
    main()
