# Comprehensive DSA Interview Patterns Guide

## Table of Contents
1. [Two Pointers](#1-two-pointers)
2. [Sliding Window](#2-sliding-window)
3. [Fast and Slow Pointers](#3-fast-and-slow-pointers)
4. [Merge Intervals](#4-merge-intervals)
5. [Cyclic Sort](#5-cyclic-sort)
6. [In-place Reversal of a LinkedList](#6-in-place-reversal-of-a-linkedlist)
7. [Tree Breadth-First Search](#7-tree-breadth-first-search)
8. [Tree Depth-First Search](#8-tree-depth-first-search)
9. [Two Heaps](#9-two-heaps)
10. [Subsets](#10-subsets)
11. [Modified Binary Search](#11-modified-binary-search)
12. [Top K Elements](#12-top-k-elements)
13. [K-way Merge](#13-k-way-merge)
14. [Topological Sort](#14-topological-sort)
15. [0/1 Knapsack](#15-01-knapsack)
16. [Fibonacci Numbers](#16-fibonacci-numbers)
17. [Palindromic Subsequence](#17-palindromic-subsequence)
18. [Longest Common Substring](#18-longest-common-substring)

## 1. Two Pointers

### Pattern Description
The Two Pointers pattern uses two pointers to iterate through the data structure in tandem until one or both of the pointers hit a certain condition. This is often used for searching pairs in a sorted array or linked list, for removing duplicates from a sorted array, or for reversing a string or linked list.

### Example Problems
1. Pair with Target Sum (Easy)
2. Remove Duplicates (Easy)
3. Squaring a Sorted Array (Easy)
4. Triplet Sum to Zero (Medium)
5. Dutch National Flag Problem (Medium)

### Template Code (Java)

```java
public class TwoPointersPattern {
    public static void twoPointers(int[] arr) {
        int left = 0;
        int right = arr.length - 1;
        
        while (left < right) {
            // Process elements from both ends
            // Move pointers based on conditions
            if (someCondition) {
                left++;
            } else {
                right--;
            }
        }
    }
}
```

### Explanation
The template starts with two pointers, `left` and `right`, at the beginning and end of the array respectively. The pointers move towards each other based on certain conditions until they meet or cross. This approach is efficient as it often reduces the time complexity from O(n^2) to O(n) for many problems.

## 2. Sliding Window

### Pattern Description
The Sliding Window pattern is used to perform a required operation on a specific window size of a given array or linked list, such as finding the longest subarray containing all 1s. It's particularly useful for problems dealing with contiguous subarrays or sublists.

### Example Problems
1. Maximum Sum Subarray of Size K (Easy)
2. Longest Substring with K Distinct Characters (Medium)
3. String Anagrams (Hard)
4. Smallest Window containing Substring (Hard)
5. Longest Substring with Distinct Characters (Hard)

### Template Code (Java)

```java
public class SlidingWindowPattern {
    public static int slidingWindow(int[] arr, int k) {
        int windowSum = 0, maxSum = 0;
        int windowStart = 0;
        
        for (int windowEnd = 0; windowEnd < arr.length; windowEnd++) {
            windowSum += arr[windowEnd]; // Add the next element
            
            // Slide the window, we don't need to slide if we've not hit the required window size of 'k'
            if (windowEnd >= k - 1) {
                maxSum = Math.max(maxSum, windowSum);
                windowSum -= arr[windowStart]; // Subtract the element going out
                windowStart++; // Slide the window ahead
            }
        }
        
        return maxSum;
    }
}
```

### Explanation
This template maintains a window of a specific size and slides it through the array. It's particularly useful for problems involving subarrays or subsequences. The window size can be fixed or variable depending on the problem. The time complexity is typically O(n) as we process each element at most twice (once when it's added to the window and once when it's removed).

## 3. Fast and Slow Pointers

### Pattern Description
The Fast and Slow Pointers pattern, also known as the Hare & Tortoise algorithm, is a pointer algorithm that uses two pointers which move through the array (or sequence/linked list) at different speeds. This approach is quite useful when dealing with cyclic linked lists or arrays.

### Example Problems
1. LinkedList Cycle (Easy)
2. Start of LinkedList Cycle (Medium)
3. Happy Number (Medium)
4. Middle of the LinkedList (Easy)
5. Palindrome LinkedList (Medium)

### Template Code (Java)

```java
public class FastSlowPointersPattern {
    public static boolean hasCycle(ListNode head) {
        if (head == null) return false;
        
        ListNode slow = head;
        ListNode fast = head;
        
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (slow == fast) {
                return true;  // Found the cycle
            }
        }
        return false;
    }
    
    class ListNode {
        int value = 0;
        ListNode next;
        ListNode(int value) {
            this.value = value;
        }
    }
}
```

### Explanation
This pattern is typically used for cycle detection in a linked list or array. The `fast` pointer moves two steps at a time while the `slow` pointer moves one step. If there's a cycle, the fast pointer will eventually meet the slow pointer. This approach has a time complexity of O(n) and space complexity of O(1).

## 4. Merge Intervals

### Pattern Description
The Merge Intervals pattern deals with problems involving overlapping intervals. In many problems involving intervals, you either need to find overlapping intervals or merge intervals if they overlap.

### Example Problems
1. Merge Intervals (Medium)
2. Insert Interval (Medium)
3. Intervals Intersection (Medium)
4. Conflicting Appointments (Medium)
5. Minimum Meeting Rooms (Hard)

### Template Code (Java)

```java
import java.util.*;

public class MergeIntervalsPattern {
    public static List<Interval> mergeIntervals(List<Interval> intervals) {
        if (intervals.size() < 2) {
            return intervals;
        }
        
        // Sort the intervals by start time
        Collections.sort(intervals, (a, b) -> Integer.compare(a.start, b.start));
        
        List<Interval> mergedIntervals = new LinkedList<Interval>();
        Iterator<Interval> intervalItr = intervals.iterator();
        Interval interval = intervalItr.next();
        int start = interval.start;
        int end = interval.end;
        
        while (intervalItr.hasNext()) {
            interval = intervalItr.next();
            if (interval.start <= end) { // Overlapping intervals, adjust the end
                end = Math.max(end, interval.end);
            } else { // Non-overlapping interval, add the previous interval and reset
                mergedIntervals.add(new Interval(start, end));
                start = interval.start;
                end = interval.end;
            }
        }
        
        // Add the last interval
        mergedIntervals.add(new Interval(start, end));
        return mergedIntervals;
    }
    
    class Interval {
        int start;
        int end;
        
        public Interval(int start, int end) {
            this.start = start;
            this.end = end;
        }
    }
}
```

### Explanation
This pattern typically involves sorting the intervals based on start time and then iterating through them to merge overlapping intervals. The time complexity is usually O(N * logN) due to the sorting step, where N is the number of intervals. The space complexity is O(N) to store the merged intervals.

## 5. Cyclic Sort

### Pattern Description
The Cyclic Sort pattern is used to deal with problems involving arrays containing numbers in a given range. It's particularly useful when the problem asks to find the missing/duplicate/smallest number in an array of numbers from 1 to n.

### Example Problems
1. Find the Missing Number (Easy)
2. Find all Missing Numbers (Easy)
3. Find the Duplicate Number (Medium)
4. Find all Duplicate Numbers (Medium)
5. Find the Corrupt Pair (Easy)

### Template Code (Java)

```java
public class CyclicSortPattern {
    public static void cyclicSort(int[] nums) {
        int i = 0;
        while (i < nums.length) {
            int correctIndex = nums[i] - 1;
            if (nums[i] != nums[correctIndex]) {
                swap(nums, i, correctIndex);
            } else {
                i++;
            }
        }
    }
    
    private static void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}
```

### Explanation
This pattern works by placing each number in its correct index. For example, if the input array is [3, 1, 5, 4, 2], then for each number 'n', we try to put it at the index 'n-1'. The time complexity is O(n) and space complexity is O(1), making it very efficient for problems with a given range of numbers.

## 6. In-place Reversal of a LinkedList

### Pattern Description
This pattern describes an efficient way to reverse a linked list. It's particularly useful when you're asked to reverse a linked list without using extra memory.

### Example Problems
1. Reverse a LinkedList (Easy)
2. Reverse a Sub-list (Medium)
3. Reverse every K-element Sub-list (Medium)
4. Reverse alternating K-element Sub-list (Medium)
5. Rotate a LinkedList (Medium)

### Template Code (Java)

```java
public class LinkedListReversalPattern {
    public static ListNode reverse(ListNode head) {
        ListNode prev = null;
        ListNode current = head;
        ListNode next = null;
        
        while (current != null) {
            next = current.next; // temporarily store the next node
            current.next = prev; // reverse the current node
            prev = current; // before we move to the next node, point prev to the current node
            current = next; // move to the next node
        }
        
        return prev; // prev is the new head
    }
    
    class ListNode {
        int value = 0;
        ListNode next;
        ListNode(int value) {
            this.value = value;
        }
    }
}
```

### Explanation
This pattern works by changing the next pointer of each node to point to its previous element. We use three pointers: `prev`, `current`, and `next` to keep track of nodes. The time complexity is O(N) where N is the number of nodes in the LinkedList. The space complexity is O(1) as we only use a constant amount of extra space.

## 7. Tree Breadth-First Search

### Pattern Description
This pattern is based on the Breadth-First Search (BFS) technique to traverse a tree and is used to solve problems involving tree traversals.

### Example Problems
1. Binary Tree Level Order Traversal (Easy)
2. Reverse Level Order Traversal (Easy)
3. Zigzag Traversal (Medium)
4. Level Averages in a Binary Tree (Easy)
5. Minimum Depth of a Binary Tree (Easy)

### Template Code (Java)

```java
import java.util.*;

public class TreeBFSPattern {
    public static List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) return result;
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        
        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            List<Integer> currentLevel = new ArrayList<>(levelSize);
            for (int i = 0; i < levelSize; i++) {
                TreeNode currentNode = queue.poll();
                currentLevel.add(currentNode.val);
                if (currentNode.left != null) {
                    queue.offer(currentNode.left);
                }
                if (currentNode.right != null) {
                    queue.offer(currentNode.right);
                }
            }
            result.add(currentLevel);
        }
        
        return result;
    }
    
    class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode(int x) { val = x; }
    }
}
```

### Explanation
BFS uses a queue to keep track of all the nodes of a level before jumping onto the next level. This is crucial in scenarios where you need to process nodes level by level. The time complexity is O(N), where N is the number of nodes in the tree, as we visit each node once. The space complexity is O(W), where W is the maximum width of the tree, which is the maximum number of nodes at any level.

## 8. Tree Depth-First Search

### Pattern Description
This pattern is based on the Depth First Search (DFS) technique to traverse a tree. It's useful for problems requiring traversing a tree in a depth-first manner, often recursively.

### Example Problems
1. Binary Tree Path Sum (Easy)
2. All Paths for a Sum (Medium)
3. Sum of Path Numbers (Medium)
4. Path With Given Sequence (Medium)
5. Count Paths for a Sum (Medium)

### Template Code (Java)

```java
public class TreeDFSPattern {
    public static boolean hasPath(TreeNode root, int sum) {
        if (root == null)
            return false;

        // if the current node is a leaf and its value is equal to the sum, we've found a path
        if (root.val == sum && root.left == null && root.right == null)
            return true;

        // recursively call to traverse the left and right sub-tree
        // return true if any of the two recursive call return true
        return hasPath(root.left, sum - root.val) || hasPath(root.right, sum - root.val);
    }
    
    class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode(int x) { val = x; }
    }
}
```

### Explanation
DFS explores as far as possible along each branch before backtracking. This is typically implemented using recursion, making it useful for problems where you need to explore all possible paths or find a path that satisfies certain conditions. The time complexity is O(N), where N is the number of nodes in the tree, as we visit each node once. The space complexity is O(H) where H is the height of the tree, due to the recursion stack.

## 9. Two Heaps (Continued)

```java
import java.util.*;

public class TwoHeapsPattern {
    PriorityQueue<Integer> maxHeap; // containing first half of numbers
    PriorityQueue<Integer> minHeap; // containing second half of numbers

    public TwoHeapsPattern() {
        maxHeap = new PriorityQueue<>((a, b) -> b - a);
        minHeap = new PriorityQueue<>((a, b) -> a - b);
    }

    public void insertNum(int num) {
        if (maxHeap.isEmpty() || maxHeap.peek() >= num)
            maxHeap.add(num);
        else
            minHeap.add(num);

        // either both the heaps will have equal number of elements or max-heap will have one
        // more element than the min-heap
        if (maxHeap.size() > minHeap.size() + 1)
            minHeap.add(maxHeap.poll());
        else if (maxHeap.size() < minHeap.size())
            maxHeap.add(minHeap.poll());
    }

    public double findMedian() {
        if (maxHeap.size() == minHeap.size()) {
            // we have even number of elements, take the average of middle two elements
            return maxHeap.peek() / 2.0 + minHeap.peek() / 2.0;
        }
        // because max-heap will have one more element than the min-heap
        return maxHeap.peek();
    }
}
```

### Explanation
This pattern uses two heaps: a max-heap to store the smaller half of the numbers and a min-heap to store the larger half. This allows us to find the median in O(1) time. The time complexity for inserting a number is O(log N) due to heap insertions, and the space complexity is O(N) to store all numbers.

## 10. Subsets

### Pattern Description
This pattern deals with problems where we need to find the combinations or permutations of a given set of elements. The pattern is based on Breadth First Search (BFS) approach.

### Example Problems
1. Subsets (Easy)
2. Subsets With Duplicates (Easy)
3. Permutations (Medium)
4. String Permutations by changing case (Medium)
5. Balanced Parentheses (Hard)

### Template Code (Java)

```java
import java.util.*;

public class SubsetsPattern {
    public static List<List<Integer>> findSubsets(int[] nums) {
        List<List<Integer>> subsets = new ArrayList<>();
        // start by adding the empty subset
        subsets.add(new ArrayList<>());
        for (int currentNumber : nums) {
            // we will take all existing subsets and insert the current number in them to create new subsets
            int n = subsets.size();
            for (int i = 0; i < n; i++) {
                // create a new subset from the existing subset and insert the current element to it
                List<Integer> set = new ArrayList<>(subsets.get(i));
                set.add(currentNumber);
                subsets.add(set);
            }
        }
        return subsets;
    }
}
```

### Explanation
This pattern generates subsets by iterating through all numbers, and for each number, creates new subsets by adding the current number to all the existing subsets. The time complexity is O(2^N), where N is the number of elements, as there are 2^N subsets. The space complexity is also O(2^N) to store all the subsets.

## 11. Modified Binary Search

### Pattern Description
This pattern describes an efficient way to handle searching in a sorted array, list, or matrix. It's particularly useful when you need to find a specific value or a range of values in a sorted data structure.

### Example Problems
1. Order-agnostic Binary Search (Easy)
2. Ceiling of a Number (Medium)
3. Next Letter (Medium)
4. Number Range (Medium)
5. Search in a Sorted Infinite Array (Medium)

### Template Code (Java)

```java
public class ModifiedBinarySearchPattern {
    public static int search(int[] arr, int key) {
        int start = 0, end = arr.length - 1;
        boolean isAscending = arr[start] < arr[end];
        while (start <= end) {
            // calculate the middle of the current range
            int mid = start + (end - start) / 2;

            if (key == arr[mid])
                return mid;

            if (isAscending) { // ascending order
                if (key < arr[mid]) {
                    end = mid - 1; // the 'key' can be in the first half
                } else { // key > arr[mid]
                    start = mid + 1; // the 'key' can be in the second half
                }
            } else { // descending order
                if (key > arr[mid]) {
                    end = mid - 1; // the 'key' can be in the first half
                } else { // key < arr[mid]
                    start = mid + 1; // the 'key' can be in the second half
                }
            }
        }
        return -1; // element not found
    }
}
```

### Explanation
This pattern uses the binary search algorithm but modifies it to handle different scenarios like ascending or descending order, or finding the ceiling or floor of a number. The time complexity is O(log N), where N is the number of elements in the array. The space complexity is O(1) as we only use a constant amount of extra space.

## 12. Top K Elements

### Pattern Description
This pattern is useful for problems where you need to find the top/smallest/frequent K elements among a given set. The pattern uses a Heap to keep track of K elements.

### Example Problems
1. Top 'K' Numbers (Easy)
2. Kth Smallest Number (Easy)
3. 'K' Closest Points to the Origin (Easy)
4. Connect Ropes (Easy)
5. Top 'K' Frequent Numbers (Medium)

### Template Code (Java)

```java
import java.util.*;

public class TopKElementsPattern {
    public static List<Integer> findKLargestNumbers(int[] nums, int k) {
        PriorityQueue<Integer> minHeap = new PriorityQueue<Integer>((n1, n2) -> n1 - n2);
        // put first 'K' numbers in the min heap
        for (int i = 0; i < k; i++)
            minHeap.add(nums[i]);

        // go through the remaining numbers of the array, if the number from the array is bigger than the
        // top (smallest) number of the min-heap, remove the top number from heap and add the number from array
        for (int i = k; i < nums.length; i++) {
            if (nums[i] > minHeap.peek()) {
                minHeap.poll();
                minHeap.add(nums[i]);
            }
        }

        // the heap has the top 'K' numbers, return them in a list
        return new ArrayList<>(minHeap);
    }
}
```

### Explanation
This pattern often uses a min-heap to keep track of the K largest elements. For K smallest elements, it uses a max-heap. The time complexity is O(N * logK) where N is the total number of elements. The space complexity is O(K) to store the heap.

## 13. K-way Merge

### Pattern Description
This pattern helps solve problems that involve merging K sorted arrays, lists, or matrices.

### Example Problems
1. Merge K Sorted Lists (Medium)
2. Kth Smallest Number in M Sorted Lists (Medium)
3. Kth Smallest Number in a Sorted Matrix (Hard)
4. Smallest Number Range (Hard)

### Template Code (Java)

```java
import java.util.*;

class ListNode {
    int value;
    ListNode next;

    ListNode(int value) {
        this.value = value;
    }
}

public class KWayMergePattern {
    public static ListNode mergeKLists(ListNode[] lists) {
        PriorityQueue<ListNode> minHeap = new PriorityQueue<>((n1, n2) -> n1.value - n2.value);

        // put the root of each list in the min heap
        for (ListNode root : lists)
            if (root != null)
                minHeap.add(root);

        // take the smallest (top) element form the min-heap and add it to the result; 
        // if the top element has a next element add it to the heap
        ListNode resultHead = null, resultTail = null;
        while (!minHeap.isEmpty()) {
            ListNode node = minHeap.poll();
            if (resultHead == null) {
                resultHead = resultTail = node;
            } else {
                resultTail.next = node;
                resultTail = resultTail.next;
            }

            if (node.next != null)
                minHeap.add(node.next);
        }

        return resultHead;
    }
}
```

### Explanation
This pattern uses a min-heap to efficiently merge K sorted lists. The heap always contains the smallest element from each of the K lists. The time complexity is O(N * logK) where N is the total number of elements in all lists, and K is the number of lists. The space complexity is O(K) for the heap.

## 14. Topological Sort

### Pattern Description
Topological Sort is used to find a linear ordering of elements that have dependencies on each other. For example, if event 'B' is dependent on event 'A', 'A' comes before 'B' in topological ordering.

### Example Problems
1. Topological Sort (Medium)
2. Tasks Scheduling (Medium)
3. Tasks Scheduling Order (Medium)
4. All Tasks Scheduling Orders (Hard)
5. Alien Dictionary (Hard)

### Template Code (Java)

```java
import java.util.*;

public class TopologicalSortPattern {
    public static List<Integer> sort(int vertices, int[][] edges) {
        List<Integer> sortedOrder = new ArrayList<>();
        if (vertices <= 0)
            return sortedOrder;

        // Initialize the graph
        HashMap<Integer, Integer> inDegree = new HashMap<>(); // count of incoming edges for every vertex
        HashMap<Integer, List<Integer>> graph = new HashMap<>(); // adjacency list graph
        for (int i = 0; i < vertices; i++) {
            inDegree.put(i, 0);
            graph.put(i, new ArrayList<Integer>());
        }

        // Build the graph
        for (int[] edge : edges) {
            int parent = edge[0], child = edge[1];
            graph.get(parent).add(child); // put the child into it's parent's list
            inDegree.put(child, inDegree.get(child) + 1); // increment child's inDegree
        }

        // Find all sources i.e., all vertices with 0 in-degrees
        Queue<Integer> sources = new LinkedList<>();
        for (Map.Entry<Integer, Integer> entry : inDegree.entrySet()) {
            if (entry.getValue() == 0)
                sources.add(entry.getKey());
        }

        // For each source, add it to the sortedOrder and subtract one from all of its children's in-degrees
        // if a child's in-degree becomes zero, add it to the sources queue
        while (!sources.isEmpty()) {
            int vertex = sources.poll();
            sortedOrder.add(vertex);
            List<Integer> children = graph.get(vertex); // get the node's children to decrement their in-degrees
            for (int child : children) {
                inDegree.put(child, inDegree.get(child) - 1);
                if (inDegree.get(child) == 0)
                    sources.add(child);
            }
        }

        if (sortedOrder.size() != vertices) // topological sort is not possible as the graph has a cycle
            return new ArrayList<>();

        return sortedOrder;
    }
}
```

### Explanation
This pattern uses the concept of in-degree (number of incoming edges) for each vertex. We start with vertices that have an in-degree of 0 (no dependencies) and gradually move to vertices with higher in-degrees as their dependencies are resolved. The time and space complexity is O(V + E), where V is the number of vertices and E is the number of edges in the graph.

## 15. 0/1 Knapsack (Dynamic Programming)

### Pattern Description
The 0/1 Knapsack pattern is based on the famous dynamic programming problem where given the weights and profits of 'N' items, we need to put these items in a knapsack with a capacity 'C'. The goal is to get the maximum profit from the items in the knapsack.

### Example Problems
1. 0/1 Knapsack (Medium)
2. Equal Subset Sum Partition (Medium)
3. Subset Sum (Medium)
4. Minimum Subset Sum Difference (Hard)
5. Count of Subset Sum (Hard)

### Template Code (Java)

```java
public class KnapsackPattern {
    public int solveKnapsack(int[] profits, int[] weights, int capacity) {
        // basic checks
        if (capacity <= 0 || profits.length == 0 || weights.length != profits.length)
            return 0;

        int n = profits.length;
        int[][] dp = new int[n][capacity + 1];

        // populate the capacity=0 columns, with '0' capacity we have '0' profit
        for(int i=0; i < n; i++)
            dp[i][0] = 0;

        // if we have only one weight, we will take it if it is not more than the capacity
        for(int c=0; c <= capacity; c++) {
            if(weights[0] <= c)
                dp[0][c] = profits[0];
        }

        // process all sub-arrays for all the capacities
        for(int i=1; i < n; i++) {
            for(int c=1; c <= capacity; c++) {
                int profit1= 0, profit2 = 0;
                // include the item, if it is not more than the capacity
                if(weights[i] <= c)
                    profit1 = profits[i] + dp[i-1][c-weights[i]];
                // exclude the item
                profit2 = dp[i-1][c];
                // take maximum
                dp[i][c] = Math.max(profit1, profit2);
            }
        }

        // maximum profit will be at the bottom-right corner.
        return dp[n-1][capacity];
    }
}
```

### Explanation
This pattern uses a 2D array to store the results of subproblems. The row 'i' represents considering items up to index 'i', and the column 'c' represents the knapsack capacity. We fill this table bottom-up. The time and space complexity is O(N*C) where 'N' is the number of items and 'C' is the knapsack capacity.

## 16. Fibonacci Numbers (Continued)

The time complexity of this approach is O(n), and the space complexity is also O(n). However, we can optimize the space complexity to O(1) by only storing the last two numbers:

```java
public class FibonacciPattern {
    public int calculateFibonacci(int n) {
        if (n < 2)
            return n;
        int n1 = 0, n2 = 1, temp;
        for (int i = 2; i <= n; i++) {
            temp = n1 + n2;
            n1 = n2;
            n2 = temp;
        }
        return n2;
    }
}
```

This optimized version maintains the O(n) time complexity but reduces the space complexity to O(1).

## 17. Palindromic Subsequence

### Pattern Description
This pattern is used to solve problems related to palindromic subsequences or substrings in a string. It typically involves using Dynamic Programming (DP) to build solutions for larger subproblems from solutions to smaller subproblems.

### Example Problems
1. Longest Palindromic Subsequence (Medium)
2. Longest Palindromic Substring (Medium)
3. Count of Palindromic Substrings (Hard)
4. Minimum Deletions in a String to make it a Palindrome (Hard)
5. Palindromic Partitioning (Hard)

### Template Code (Java)

```java
public class PalindromicSubsequencePattern {
    public int findLPSLength(String st) {
        int n = st.length();
        int[][] dp = new int[n][n];
        
        // every sequence with one element is a palindrome of length 1
        for (int i = 0; i < n; i++)
            dp[i][i] = 1;
            
        for (int startIndex = n - 1; startIndex >= 0; startIndex--) {
            for (int endIndex = startIndex + 1; endIndex < n; endIndex++) {
                // case 1: elements at the beginning and the end are the same
                if (st.charAt(startIndex) == st.charAt(endIndex)) {
                    dp[startIndex][endIndex] = 2 + dp[startIndex + 1][endIndex - 1];
                } else { // case 2: skip one element either from the beginning or the end
                    dp[startIndex][endIndex] = Math.max(dp[startIndex + 1][endIndex], dp[startIndex][endIndex - 1]);
                }
            }
        }
        return dp[0][n - 1];
    }
}
```

### Explanation
This pattern uses a 2D DP table where `dp[i][j]` represents the length of the longest palindromic subsequence from index i to j in the string. We fill this table bottom-up, starting with substrings of length 1 and gradually increasing the substring length. The time and space complexity is O(n^2) where n is the length of the string.

## 18. Longest Common Substring

### Pattern Description
This pattern is used to find the longest common substring or subsequence between two strings. It's another classic Dynamic Programming problem.

### Example Problems
1. Longest Common Substring (Medium)
2. Longest Common Subsequence (Medium)
3. Minimum Deletions & Insertions to Transform a String into another (Hard)
4. Longest Increasing Subsequence (Medium)
5. Maximum Sum Increasing Subsequence (Hard)

### Template Code (Java)

```java
public class LongestCommonSubstringPattern {
    public int findLCSLength(String s1, String s2) {
        int[][] dp = new int[s1.length() + 1][s2.length() + 1];
        int maxLength = 0;
        
        for (int i = 1; i <= s1.length(); i++) {
            for (int j = 1; j <= s2.length(); j++) {
                if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
                    dp[i][j] = 1 + dp[i - 1][j - 1];
                    maxLength = Math.max(maxLength, dp[i][j]);
                }
            }
        }
        return maxLength;
    }
}
```

### Explanation
This pattern uses a 2D DP table where `dp[i][j]` represents the length of the longest common substring ending at index i-1 in s1 and j-1 in s2. We fill this table iteratively, and keep track of the maximum length seen so far. The time and space complexity is O(m*n) where m and n are the lengths of the two strings.

## Additional Tips for Coding Interviews

1. **Clarify the Problem**: Always start by clarifying the problem. Ask questions about input size, data types, edge cases, and expected output.

2. **Think Aloud**: Communicate your thought process. Interviewers want to know how you approach problems.

3. **Start with Brute Force**: If you can't immediately see an optimal solution, start with a brute force approach and then optimize.

4. **Optimize**: Look for ways to optimize your solution. Can you reduce time complexity? Space complexity?

5. **Test Your Code**: Before saying you're done, test your code with a few test cases, including edge cases.

6. **Time and Space Complexity**: Always be prepared to discuss the time and space complexity of your solution.

7. **Code Clarity**: Write clean, well-organized code. Use meaningful variable names and add comments if necessary.

8. **Handle Edge Cases**: Don't forget to handle edge cases like null inputs, empty arrays/strings, etc.

9. **Use Built-in Functions Wisely**: Know your language's standard library, but be prepared to implement things from scratch if asked.

10. **Practice, Practice, Practice**: The more problems you solve, the more patterns you'll recognize, and the better prepared you'll be.

Remember, these patterns are guidelines, not strict rules. Real interview questions often combine multiple patterns or require novel approaches. The key is to practice applying these patterns to various problems so you can recognize when and how to use them in an interview setting.

