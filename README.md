# 暑假每日一题

### [两数之和 - 2023/7/1](https://leetcode.cn/problems/two-sum/)

```c++
class Solution {
public:
    map<int,int> vis;
    vector<int> twoSum(vector<int>& nums, int target) {
        for (int i = 0; i < nums.size(); ++ i) {
            if (vis.count(target - nums[i])) {
                return {vis[target - nums[i]], i};
            } 
            vis[nums[i]] = i;
        }
        return {};
    }
};
```

### [两数相加 - 2023/7/2](https://leetcode.cn/problems/add-two-numbers/)

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* res = new ListNode();
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        dfs(res, l1, l2);
        return res->next;
    }

    void dfs(ListNode* fa, ListNode* l1, ListNode* l2) {
        if (l1 == nullptr && l2 == nullptr) {
            // 最后以为需要进位的情况
            if (fa->val >= 10) {
                ListNode* node = new ListNode(1);
                fa->val %= 10;
                fa->next = node;
            }
            return;
        }
        int n1 = 0, n2 = 0;
        if (l1 != nullptr) {
            n1 = l1->val;
        }
        if (l2 != nullptr) {
            n2 = l2->val;
        }
        int res = n1 + n2;
        ListNode* node = new ListNode();
        // 父亲结点是否超出的情况
        node->val = res + (fa->val >= 10 ? 1 : 0);
        // 父亲结点的变化
        fa->val %= 10;
        fa->next = node;
        // 遍历的三种情况
        if (l1 == nullptr) {
            dfs(node, l1, l2->next);
        } else if (l2 == nullptr) {
            dfs(node, l1->next, l2);
        } else dfs(node, l1->next, l2->next);
    }
};
```

