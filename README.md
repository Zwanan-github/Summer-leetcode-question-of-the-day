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

### [两数相加 II - 2023/7/3](https://leetcode.cn/problems/add-two-numbers-ii/submissions/)

#### 解法一

深搜到底部，然后回溯

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

    int len1 = 0, len2 = 0;

    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode* a = l1;
        ListNode* b = l2;
        while (a != nullptr) {
            a = a->next;
            len1++;
        }
        while (b != nullptr) {
            b = b->next;
            len2++;
        }
        ListNode* node = dfs(l1, l2, len1, len2);
        // 对最后头元素大于10特判
        ListNode* res = new ListNode(1);;
        if (node->val >= 10) {
            node->val %= 10;
            res->next = node;
            return res;
        } 
        return node;
    }
	// 深搜回溯
    ListNode* dfs(ListNode* l1, ListNode* l2, int idx1, int idx2) {
        ListNode* node = new ListNode();
        if (idx1 == 0 && idx2 == 0) return nullptr;
        else if (idx1 > idx2) {
            int sum = l1->val;
            ListNode* x = dfs(l1->next, l2, idx1 - 1, idx2);
            node->val = sum;
            if (x != nullptr && x->val >= 10) {
                x->val %= 10;
                node->val += 1;
            }
            node->next = x;
        }
        else if (idx1 < idx2) {
            int sum = l2->val;
            ListNode* x = dfs(l1, l2->next, idx1, idx2 - 1);
            node->val = sum;
            if (x != nullptr && x->val >= 10) {
                x->val %= 10;
                node->val += 1;
            }
            node->next = x;
        }
        else if (idx1 == idx2) {
            int sum = l1->val + l2->val;
            ListNode* x = dfs(l1->next, l2->next, idx1 - 1, idx2 - 1);
            node->val = sum;
            if (x != nullptr && x->val >= 10) {
                x->val %= 10;
                node->val += 1;
            }
            node->next = x;
        }
        return node;
    }

};
```

#### 解法二

使用栈来逆向遍历

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
const int N = 110;

class Solution {
public:

    int stack1[N], stack2[N];
    int h1 = 0, t1 = -1, h2 = 0, t2 = -1;
    ListNode* res = nullptr;

    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        while (l1 != nullptr) {
            stack1[++t1] = l1->val;
            l1 = l1->next;
        }
        while (l2 != nullptr) {
            stack2[++t2] = l2->val;
            l2 = l2->next;
        }
        while (h1 <= t1 && h2 <= t2) {
            int sum = stack1[t1--] + stack2[t2--];
            ListNode* node = new ListNode(sum);
            if (res == nullptr) {
                res = node;
            } else {
                if (res->val >= 10) {
                    node->val += 1;
                    res->val %= 10;
                    node->next = res;
                    res = node;
                } else {
                    node->next = res;
                    res = node;
                }
            }
        }
        while (h1 <= t1) {
            int sum = stack1[t1--];
            ListNode* node = new ListNode(sum);
            if (res == nullptr) {
                res = node;
            } else {
                if (res->val >= 10) {
                    node->val += 1;
                    res->val %= 10;
                    node->next = res;
                    res = node;
                } else {
                    node->next = res;
                    res = node;
                }
            }
        }
        while (h2 <= t2) {
            int sum = stack2[t2--];
            ListNode* node = new ListNode(sum);
            if (res == nullptr) {
                res = node;
            } else {
                if (res->val >= 10) {
                    node->val += 1;
                    res->val %= 10;
                    node->next = res;
                    res = node;
                } else {
                    node->next = res;
                    res = node;
                }
            }
        }
        ListNode* node = new ListNode(0);
        if (res->val >= 10) {
            node->val += 1;
            res->val %= 10;
            node->next = res;
            res = node;
            return res;
        } 
        return res;
    }
};
```

### [ 矩阵中的和 - 2023/7/4](https://leetcode.cn/problems/sum-in-a-matrix/submissions/)

```c++
class Solution {
public:
    int matrixSum(vector<vector<int>>& nums) {
        int res = 0;
        // 对每行sort(), O(n*mlogm)
        for (int i = 0; i < nums.size(); ++ i) {
            sort(nums[i].begin(), nums[i].end());
        }
        // multiset的操作都是logn
        multiset<int> set;
        // n * m * logn
        for (int i = 0; i < nums[0].size(); ++ i) {
            set.clear();
            for (int j = 0; j < nums.size(); ++ j) {
                set.insert(nums[j][i]);
            }
            res += *set.rbegin();
        }
        return res;
    }
};
```

### [K 件物品的最大和 - 2023/7/5](https://leetcode.cn/problems/k-items-with-the-maximum-sum/description/)

```c++
class Solution {
public:
    int kItemsWithMaximumSum(int numOnes, int numZeros, int numNegOnes, int k) {
        return k >= numOnes ? numOnes + (k - numOnes >= numZeros ? -(k-numOnes-numZeros) : 0) : k;
    }
};
```

### [拆分成最多数目的正偶数之和 - 2023/7/6](https://leetcode.cn/problems/maximum-split-of-positive-even-integers/)

```c++
class Solution {
public:
    vector<long long> maximumEvenSplit(long long finalSum) {
        vector<long long> ans;
        if (finalSum % 2 == 1) {
            return ans;
        }
        long long bit = 2;
        while (finalSum >= bit) {
            finalSum -= bit;
            ans.push_back(bit);
            bit += 2;
        }
        if (finalSum) {
            long long last = *ans.rbegin();
            ans.pop_back();
            ans.push_back(finalSum + last);
        }
        return ans;
    }
};
```

### [过桥的时间 - 2023/7/7](https://leetcode.cn/problems/time-to-cross-a-bridge/description/)

大模拟，得把题摸清楚规则

```c++
class Solution {
public:
    using PII = pair<int, int>;
    int findCrossingTime(int n, int k, vector<vector<int>>& time) {
        // 定义等待中的工人优先级比较规则，时间总和越高，效率越低，优先级越低，越优先被取出
        auto wait_priority_cmp = [&](int x, int y) {
            int time_x = time[x][0] + time[x][2];
            int time_y = time[y][0] + time[y][2];
            return time_x != time_y ? time_x < time_y : x < y;
        };

        priority_queue<int, vector<int>, decltype(wait_priority_cmp)> wait_left(wait_priority_cmp), wait_right(wait_priority_cmp);

        priority_queue<PII, vector<PII>, greater<PII>> work_left, work_right;

        int remain = n, cur_time = 0;
        for (int i = 0; i < k; i++) {
            wait_left.push(i);
        }
        while (remain > 0 || !work_right.empty() || !wait_right.empty()) {
            // 1. 若 work_left 或 work_right 中的工人完成工作，则将他们取出，并分别放置到 wait_left 和 wait_right 中。
            while (!work_left.empty() && work_left.top().first <= cur_time) {
                wait_left.push(work_left.top().second);
                work_left.pop();
            }
            while (!work_right.empty() && work_right.top().first <= cur_time) {
                wait_right.push(work_right.top().second);
                work_right.pop();
            }

            if (!wait_right.empty()) {
                // 2. 若右侧有工人在等待，则取出优先级最低的工人并过桥
                int id = wait_right.top();
                wait_right.pop();
                // 加上过桥时间
                cur_time += time[id][2];
                // 把在左边放下东西的时间存入，整段时间
                work_left.push({cur_time + time[id][3], id});
            } else if (remain > 0 && !wait_left.empty()) {
                // 3. 若右侧还有箱子，并且左侧有工人在等待，则取出优先级最低的工人并过桥
                int id = wait_left.top();
                wait_left.pop();
                // 加上当前过桥的时间
                cur_time += time[id][0];
                // 把拿完东西之后的时间存在右边的工作了的序列里
                work_right.push({cur_time + time[id][1], id});
                remain--;
            } else {
                // 4. 否则，没有人需要过桥，时间过渡到 work_left 和 work_right 中的最早完成时间
                int next_time = INT_MAX;
                if (!work_left.empty()) {
                    next_time = min(next_time, work_left.top().first);
                }
                if (!work_right.empty()) {
                    next_time = min(next_time, work_right.top().first);
                }
                if (next_time != INT_MAX) {
                    cur_time = max(next_time, cur_time);
                }
            }
        }
        return cur_time;
    }
};
```



# 周赛

## 第352场周赛-2023/7/2

### [灵神的题解](https://www.bilibili.com/video/BV1ej411m7zV/?vd_source=948c0cef7c69fc77317e4c2a454ea6c9)

### [6909. 最长奇偶子数组](https://leetcode.cn/problems/longest-even-odd-subarray-with-threshold/submissions/)

```c++
class Solution {
public:
    vector<int> modRes = vector(110, 0);
    int longestAlternatingSubarray(vector<int>& nums, int threshold) {
        int l = 0;
        int flag = 0;
        for (int i = 0; i < nums.size(); ++ i) {
            modRes[i] = nums[i] % 2;
            if (!flag && modRes[i] == 0) {
                l = i;
                flag = 1;
            }
        }
        flag = 0;
        int res = 0, i = l;
        while (i < nums.size()) {
            if (modRes[i] != flag || nums[i] > threshold) {
                res = max(res, i - l);
                if (modRes[i] == flag && nums[i] > threshold) ++i;                
                while (i < nums.size() && modRes[i] != 0) {
                    ++i;
                }
                l = i;
                flag = 0;
            } else {
                res = max(res, i - l + 1);
                ++ i;
                flag = !flag;
            }
        } 
        return res;
    }
};
```

### [6916. 和等于目标值的质数对](https://leetcode.cn/problems/prime-pairs-with-target-sum/)

```c++
const int N = 1000010;

class Solution {
public:
    int st[N];
    int primes[N];
    int cnt = 0;
    // 欧拉筛
    void get_primes(int n) {
        for (int i = 2; i <= n; ++i) {
            if (!st[i]) {
                primes[cnt++] = i;
            } 
            for (int j = 0; primes[j] <= n / i; ++j) {
                st[primes[j] * i] = 1;
                if (i % primes[j] == 0) break;
            }
        }
    }
    vector<vector<int>> findPrimePairs(int n) {
        vector<vector<int>> res;
        get_primes(n);
        for (int i = 0; i < cnt; ++ i) {
            if (primes[i] <= n / 2 && !st[primes[i]] && !st[n - primes[i]]) {
                vector<int> v;
                v.push_back(primes[i]);
                v.push_back(n - primes[i]);
                res.push_back(v);
            }
        }
        return res;
    }
};
```

### [6911. 不间断子数组](https://leetcode.cn/problems/continuous-subarrays/)

```c++
class Solution {
public:
    long res = 0;
    // 有序不去重的set
    multiset<int> set; 
    long long continuousSubarrays(vector<int>& nums) {
        // 双指针遍历
        for (int i = 0, j = 0; i < nums.size(); ++ i) {
            set.insert(nums[i]);
            // 当前有序set中取头尾差值
            while (j <= i && *set.rbegin() - *set.begin() > 2) {
                // 删除头
                set.erase(set.find(nums[j]));
                // 向右移
                ++j;
            }
            res += i - j + 1;
        }
        return res;
    }
};
```

#### 类似题目

#### [1438. 绝对差不超过限制的最长连续子数组](https://leetcode.cn/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/)

```c++
class Solution {
public:
    int res = 0;
    multiset<int> set;
    int longestSubarray(vector<int>& nums, int limit) {
        for (int i = 0, j = 0; i < nums.size(); ++ i) {
            set.insert(nums[i]);
            while (j <= i && *set.rbegin() - *set.begin() > limit) {
                set.erase(set.find(nums[j]));
                ++j;
            }
            res = max(res, i - j + 1);
        }
        return res;
    }
};
```

### [6894. 所有子数组中不平衡数字之和](https://leetcode.cn/problems/sum-of-imbalance-numbers-of-all-subarrays/)

#### 解法一

$O(n^2)$双指针解法

```c++
class Solution {
public:
    int sumImbalanceNumbers(vector<int>& nums) {
        int n = nums.size();
        int ans = 0;
        int vis[n + 5];
        memset(vis, -1, sizeof vis);
        for (int i = 0; i < n; ++ i) {
            vis[nums[i]] = i;
            int cnt = 0;
            // 以nums[i] 开头的子数组
            for (int j = i + 1; j < n; ++ j) {
                // 没遍历过
                if (vis[nums[j]] != i) {
                    // 如果 nums[j] - 1 || nums[j] + 1 出现过，则减一
                    cnt += 1 - (vis[nums[j] - 1] == i) - (vis[nums[j] + 1] == i);
                    // 遍历过了
                    vis[nums[j]] = i;
                }
                ans += cnt;
            }
            
        }
        return ans;
    }
};
```

#### 解法二

$O(n)$贡献法（依旧不太会）

```c++
class Solution {
public:
    int sumImbalanceNumbers(vector<int> &nums) {
        int n = nums.size(), right[n], idx[n + 1];
        fill(idx, idx + n + 1, n);
        for (int i = n - 1; i >= 0; i--) {
            int x = nums[i];
            // right[i] 表示 nums[i] 右侧的 x 和 x-1 的最近下标（不存在时为 n）
            right[i] = min(idx[x], idx[x - 1]);
            idx[x] = i;
        }

        int ans = 0;
        memset(idx, -1, sizeof(idx));
        for (int i = 0; i < n; i++) {
            int x = nums[i];
            // 统计 x 能产生多少贡献
            ans += (i - idx[x - 1]) * (right[i] - i); // 子数组左端点个数 * 子数组右端点个数
            idx[x] = i;
        }
        // 上面计算的时候，每个子数组的最小值必然可以作为贡献，而这是不合法的
        // 所以每个子数组都多算了 1 个不合法的贡献
        return ans - n * (n + 1) / 2;
    }
};
```

