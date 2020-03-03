
#Write a Python function that takes a sequence of numbers and determines whether all the numbers are different from each other.

def determine_equal_or_not(seq_of_numbers):

    for element in seq_of_numbers:
        counter = 0
        for i in seq_of_numbers:
            if element is i:
                counter = counter + 1
        #print("element "+ element + " apears "+ counter + "x")
        print(counter)

#determine_equal_or_not([1,1,3])

#Write a Python program to create all possible strings by using 'a', 'e', 'i', 'o', 'u'. Use the characters exactly once.
from itertools import permutations
def string_permutations():
    vowel = permutations(["a","e","i","o","u"])
    for i in list(vowel):
        print(i)

#Write a Python program to remove and print every third number from a list of numbers until the list becomes empty.

def remove_nums(int_list):
  #list starts with 0 index
  position = 3 - 1 
  idx = 0
  len_list = (len(int_list))
  while len_list>0:
    #print((position+idx)%len_list)
    idx = (position+idx)%len_list
    print(int_list.pop(idx))
    len_list -= 1
nums = [10,20,30,40,50,60,70,80,90]
#remove_nums(nums)

#Write a Python program to print a long text, convert the string to a list and print all the words and their frequencies.
string_words = 'a a a a b b. bb bb. c, d, c'

word_list = string_words.split()

word_freq = [word_list.count(n) for n in word_list]

print("String:\n {} \n".format(string_words))
print("List:\n {} \n".format(str(word_list)))
print("Pairs (Words and Frequencies:\n {}".format(str(list(zip(word_list, word_freq)))))

#my solution: 
for word in set(word_list):
    n = word_list.count(word)
    print(word, n)
print("\n\n")
#Write a Python program to count the number of each character of a given text of a text file.
string = "abc def ab ab abc"
char_list = list(string)
for c in set(char_list):
    n = char_list.count(c)
    print(c, n)

#Write a Python program to find the number of divisors of a given integer is even or odd.
integer = input("Integer:  ")
integer = int(integer)
if integer % 2 == 0:
    print("Integer " + str(integer) + " is even.")
else:
     print("Integer " + str(integer) + " is odd.")
divider_list=[]
for i in range(1,integer):
    if integer % i == 0:
        divider_list.append(i)
print(divider_list)

#47 Write a Python program which reads a text (only alphabetical characters and spaces.) and prints two words. The first one is the <<word which is arise most frequently>> in the text. The second one is the <<word which has the maximum number of letters>>

string_words = 'Hey du ! Du hast mich vergessen.'
word_list = string_words.lower().split()
max_length = 1
longest_word = ''

for word in set(word_list):
    n = word_list.count(word)
    print(word, n, len(word))
     
    if len(word) > max_length:
        max_length = len(word)
        longest_word = word
print("longest word is '"+ longest_word + "' with "+ str(max_length)+ " chars.")
print("\n\n")
        