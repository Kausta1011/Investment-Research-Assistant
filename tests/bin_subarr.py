nums = [1, 1, 0, 1, 1, 0, 0, 1]
goal = 2

l = 0
no_of_arr = 0
arr_sum = 0
freq = {}

for r in range(len(nums)):
    print(r)
    if nums[r] in freq:
        freq[nums[r]] += 1
    else:
        freq[nums[r]] = 1
    
    print('count of 1:',freq.get(1))

    while freq.get(1) > goal:
        freq[nums[l]] -= 1
        l+=1
        print('count of 1 after reducing:',freq.get(1))
    if freq.get(1) == goal:
        no_of_arr += 1
    print('No_of arr:',no_of_arr)
    print()

print(no_of_arr) 


