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

### [两数之和 II - 输入有序数组 - 2023/7/8](https://leetcode.cn/problems/two-sum-ii-input-array-is-sorted/)

```c++
class Solution {
public:
    map<int,int> vis;
    vector<int> twoSum(vector<int>& nums, int target) {
        for (int i = 0; i < nums.size(); ++ i) {
            if (vis.count(target - nums[i])) {
                return {vis[target - nums[i]] + 1, i + 1};
            } 
            vis[nums[i]] = i;
        }
        return {};
    }
};
```



### [三数之和 - 2023/7/9](https://leetcode.cn/problems/3sum/)

#### 暴力法

```c++
class Solution {
public:
    int search(vector<int>& nums, int l, int x) {
        int r = nums.size();
        while (l < r) {
            int mid = (l + r) >> 1;
            if (nums[mid] < x) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }
        if (l >= nums.size() || nums[l] != x) return -1;
        else return l;
    }

    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> res;
        int n = nums.size();
        sort(nums.begin(), nums.end());
        for (int i = 0; i < n; ++ i) {
            if (i != 0 && nums[i] == nums[i - 1]) continue;
            for (int j = i + 1; j < n; ++ j) {
                if (j != i + 1 && nums[j] == nums[j - 1]) continue;
                int idx = search(nums, j + 1, 0 - nums[i] - nums[j]);
                if (idx != -1 && idx < nums.size()) {
                    res.push_back({nums[i],nums[j],0 - nums[i] - nums[j]});
                }
            }
        }
        return res;
    }
};
```

#### 双指针法

```c++
class Solution {
public:

    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> res;
        int n = nums.size();
        sort(nums.begin(), nums.end());
        for (int i = 0; i < n; ++ i) {
            if (i != 0 && nums[i] == nums[i - 1]) continue;
            for (int j = i + 1, k = n - 1; j < n; ++ j) {
                if (j != i + 1 && nums[j] == nums[j - 1]) continue;
                while (j < k && nums[j] + nums[k] + nums[i] > 0) --k;
                // 结束
                if (j == k) break;
                if (nums[i] + nums[j] + nums[k] == 0) {
                    res.push_back({nums[i], nums[j], nums[k]});
                }
            }
        }
        return res;
    }
};
```



### [最接近的三数之和 - 2023/7/10](https://leetcode.cn/problems/3sum-closest/)

```c++
class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        int res = 0x3f3f3f3f;
        sort(nums.begin(), nums.end());
        int n = nums.size();
        for (int i = 0; i < n; ++ i) {
            // 双指针二分
            for (int j = i + 1, k = n - 1; j < k;) {
                int sum = nums[i] + nums[j] + nums[k];
                if (abs(target - sum) < abs(target - res)) {
                    res = sum;
                } 
                if (sum > target) {
                    -- k;
                } else if (sum < target) {
                    ++ j;
                } else return res;
            }
        }
        return res;
    }
};
```



### [最大子序列交替和 - 2023/7/11](https://leetcode.cn/problems/maximum-alternating-subsequence-sum/)

#### 朴素动态规划

```c++
 const int N = 100010;
 class Solution {
 public:
     long long g[N], f[N];
     long long maxAlternatingSum(vector<int>& nums) {
         int n = nums.size();
         for (int i = 1; i <= n; ++ i) {
             int& x = nums[i - 1];
             // 当前在 i 个中挑选奇数个 => 在 i - 1 个中挑选偶数的最大 - 当前数字
             f[i] = max(g[i - 1] - x, f[i - 1]);
             // 当前在 i 个中挑选偶数个 => 在 i - 1 个中挑选奇数的最大 + 当前数字
             g[i] = max(f[i - 1] + x, g[i - 1]);
         }
         return max(f[n], g[n]);
     }
 };
```

#### 滑动数组，压缩内存

```c++
 class Solution {
 public:
     long long maxAlternatingSum(vector<int>& nums) {
         int n = nums.size();
         long long even = 0, odd = 0; 
         long long _even = even, _odd = odd;
         for (int i = 1; i <= n; ++ i) {
             _even = even, _odd = odd;
             int& cur = nums[i - 1];
             // 当前在 i 个中挑选奇数个 => 在 i - 1 个中挑选偶数的最大 - 当前数字
             odd = max(_odd, _even - cur);
             // 当前在 i 个中挑选偶数个 => 在 i - 1 个中挑选奇数的最大 + 当前数字
             even = max(_even, _odd + cur);
         }
         return max(even, odd);
     }
 };
```

### [交替数字和 - 2023/7/12](https://leetcode.cn/problems/alternating-digit-sum/description/)

```c++
class Solution {
public:
    int alternateDigitSum(int n) {
        int flag = 1, sum = 0, cnt = 0;
        while (n != 0) {
            sum += flag * (n % 10);
            n /= 10;
            flag = -flag;
            ++ cnt;
        }
        return sum * (cnt % 2 == 0 ? -1 : 1) ;
    }
};
```

### [下降路径最小和 - 2023/7/13](https://leetcode.cn/problems/minimum-falling-path-sum/description/)

```c++
class Solution {
public:
    int minFallingPathSum(vector<vector<int>>& matrix) {
        int n = matrix.size(), res = INT_MAX;
        int f[n][n];
        for (int i = 0; i < n; ++ i) 
            for (int j = 0; j < n; ++ j) 
                f[i][j] = INT_MAX;
        for (int i = 0; i < n; ++ i) {
            f[0][i] = matrix[0][i];
        }
        for (int i = 1; i < n; ++ i) {
            for (int j = 0; j < n; ++ j) {
                if (j > 0) f[i][j] = min(f[i][j], f[i - 1][j - 1] + matrix[i][j]);               
                if (j < n - 1) f[i][j] = min(f[i][j], f[i - 1][j + 1] + matrix[i][j]);
                f[i][j] = min(f[i][j], f[i - 1][j] + matrix[i][j]);
            }
        }
        for (int i = 0; i < n; ++ i) {
            res = min(res, f[n - 1][i]);
        } 
        return res;
    }
};
```

### [在二叉树中分配硬币 - 2023/7/14](https://leetcode.cn/problems/distribute-coins-in-binary-tree/description/)

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int ans = 0;

    int distributeCoins(TreeNode* root) {
        dfs(root);
        return ans;
    }

    pair<int,int> dfs(TreeNode* root) {
        if (root == nullptr) return {0, 0};
        auto left = dfs(root->left);
        auto right = dfs(root->right);
        // 该子树的硬币数
        int c = left.first + right.first + root->val;
        // 该子树的结点数
        int n = left.second + right.second + 1;
        // 计算从该两条边应该累加走几次
        ans += abs(c - n);
        return {c, n};
    }
};
```

### [四数之和 - 2023/7/15](https://leetcode.cn/problems/4sum/description/)

``` c++
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        int n = nums.size();
        sort(nums.begin(), nums.end());
        vector<vector<int>> ans;
        for (int a = 0; a < n; ++ a) {
            if (a != 0 && nums[a] == nums[a - 1]) continue;
            for (int b = a + 1; b < n; ++ b) {
                if (b != a + 1 && nums[b] == nums[b - 1]) continue;
                int c = b + 1, d = n - 1;
                while (c < d) {
                    long long res = (long long)target - nums[a] - nums[b] - nums[c] - nums[d];
                    if (res == 0) {
                        // 先过滤重复
                        while (c < d && nums[c] == nums[c + 1]) ++c;
                        while (c < d && nums[d] == nums[d - 1]) --d;
                        ans.push_back({nums[a], nums[b], nums[c], nums[d]}); 
                        ++c, --d; 
                    } else if (res > 0) {
                        ++c;
                    } else --d;
                }
            }
        }
        return ans;
    }
};
```

### **[树中距离之和 - 2023/7/16](https://leetcode.cn/problems/sum-of-distances-in-tree/description/)

树上dp

题解：[834. 树中距离之和 - 力扣（Leetcode）](https://leetcode.cn/problems/sum-of-distances-in-tree/solutions/2345592/tu-jie-yi-zhang-tu-miao-dong-huan-gen-dp-6bgb/)

```c++
class Solution {
public:

    vector<int> sumOfDistancesInTree(int n, vector<vector<int>>& edges) {
        // 无向图
        vector<vector<int>> e(n);
        vector<int> size(n, 1);
        vector<int> ans(n);
        for (auto& x : edges) {
            e[x[0]].push_back(x[1]);
            e[x[1]].push_back(x[0]);
        }
        // 预处理ans[0],和size
        function<void(int,int,int)> dfs = [&](int u, int fa, int depth){
            ans[0] += depth;
            for (int& x : e[u]) {
                if (x != fa) {
                    dfs(x, u, depth + 1);
                    size[u] += size[x];
                }
            }
        };
        dfs(0, -1, 0);
        function<void(int,int)> dfs2 = [&](int u, int fa) {
            for (int& x : e[u]) {
                if (x != fa) {
                    ans[x] = ans[u] + n - 2 * size[x]; 
                    dfs2(x, u);
                }
            }
        };
        dfs2(0, -1);
        return ans;
    }
};
```

### [字符串相加 - 2023/7/17](https://leetcode.cn/problems/add-strings/description/)

高精度加法

```c++
class Solution {
public:
    string addStrings(string num1, string num2) {
        if (num1.size() < num2.size()) return addStrings(num2, num1);
        int acc = 0;
        for (int i = num1.size() - 1, j = num2.size() - 1; i >= 0; -- i , --j) {
            int n1 = num1[i] - '0';
            if (acc) {
                n1 += 1;
                acc = 0;
            }
            if (j >= 0) {
                int n2 = num2[j] - '0';
                n1 += n2;
            }
            if (n1 >= 10) {
                acc = 1;
                n1 %= 10;
            }
            num1[i] = (n1 + '0');
        }
        if (acc) {
            num1 = "1" + num1;
        }
        return num1;
    }
};
```

### [包含每个查询的最小区间 - 2023/7/18](https://leetcode.cn/problems/minimum-interval-to-include-each-query/description/)

```c++
class Solution {
public:
    vector<int> minInterval(vector<vector<int>>& itv, vector<int>& queries) {
        int n = queries.size();
        // 最小堆
        multimap<int, int> map;
        for (int i = 0; i < n; ++ i) {
            map.insert({queries[i], i});
        }
        vector<int> ans(n, -1);
        // 按照区间长度走，满足条件的走完直接在multimap中删除
        sort(itv.begin(), itv.end(), [](auto& a, auto& b) {
            return a[1]- a[0] < b[1] - b[0]; 
        });
        for (auto& itvs : itv) {
            int start = itvs[0], end = itvs[1];
            for (auto it = map.lower_bound(start); it != map.end() && it->first <= end; it = map.erase(it)) {
                ans[it->second] = end - start + 1;
            }
        }
        return ans;
    }
};
```

### [模拟行走机器人 - 2023/7/19](https://leetcode.cn/problems/walking-robot-simulation/)

```c++
class Solution {
public:
    
    int robotSim(vector<int>& commands, vector<vector<int>>& obstacles) {
        int dx[] = {0,1,0,-1}, dy[] = {1, 0, -1, 0};
        int idx = 0, ans = 0;
        pair<int, int> dis = {0, 0};
        multiset<pair<int,int>> set;
        for (auto&& obstacle : obstacles) {
            set.insert({obstacle[0], obstacle[1]});
        }
        for (int& command : commands) {
            if (command == -2) {
                idx = (idx - 1) < 0 ? 3 : idx - 1;
            } else if (command == -1) {
                idx = (idx + 1) > 3 ? 0 : idx + 1;
            } else if (command > 0) {
                while (command-- && set.count({dis.first + dx[idx], dis.second + dy[idx]}) == 0) {
                    dis.first += dx[idx];
                    dis.second += dy[idx];
                    ans = max(ans, dis.first * dis.first + dis.second * dis.second);
                }
                cout << dis.first << " " << dis.second << '\n';
            }
        }
        
        return ans;
    }
};
```

### **[918. 环形子数组的最大和 - 2023/7/20](https://leetcode.cn/problems/maximum-sum-circular-subarray/description/)

```c++
class Solution {
public:
    int maxSubarraySumCircular(vector<int>& nums) {
        int n = nums.size();
        int a[n * 2 + 1];
        for (int i = 1; i <= n; ++ i) {
            a[i] = a[i + n] = nums[i - 1];
        }
        int ans = -0x3f3f3f3f;
        for (int i = 1; i <= n * 2; ++ i) {
            a[i] += a[i - 1];
        } 
        int f[2 * n + 1], h = 0, t = 0;
        // 先把前缀和头放入
        f[0] = 0;
        for (int i = 1; i <= n * 2; ++ i) {
            while (h <= t && f[t] - f[h] + 1 > n) ++h;
            // 求当前数结尾的单调队列的最大前缀和
            ans = max(ans, a[i] - a[f[h]]);
            // 把小的放进来，后面的权重就更大
            while (h <= t && a[f[t]] >= a[i]) --t;
            f[++t] = i;
        }
        return ans;
    }
};
```

### [满足不等式的最大值 - 2023/7/21](https://leetcode.cn/problems/max-value-of-equation/description/)

```c++
class Solution {
public:
    int findMaxValueOfEquation(vector<vector<int>>& points, int k) {
        int n = points.size();
        int f[n + 1][2], h = 0, t = -1, ans = -0x3f3f3f3f;
        for (int i = 0; i < n; ++ i) {
            int x = points[i][0], y = points[i][1];
            while (h <= t && x - f[h][0] > k) ++h;
            // cout << x << " " << f[h][0] << '\n';
            if (h <= t) ans = max(ans, x + y + f[h][1]);
            // cout << ans << " " << x << " " << y - x << '\n';
            while (h <= t && y - x >= f[t][1]) --t;
            f[++t][0] = x, f[t][1] = y - x;
        }
        return ans;
    }
};
```

### [柠檬水找零 - 2023/7/22](https://leetcode.cn/problems/lemonade-change/description/)

```c++
class Solution {
public:
    bool lemonadeChange(vector<int>& bills) {
        int f[4];
        memset(f, 0, sizeof f);
        for (int bill : bills) {
            int t = bill;
            t -= 5;
            for (int i = 3; i >= 0; -- i) {
                if (t == 0) break;
                // 跳过15的情况
                if (i == 2) continue;
                if (t >= (i + 1) * 5) {
                    int cnt = t / ((i + 1) * 5);
                    if (cnt <= f[i]) {
                        f[i] -= cnt;
                        t -= ((i + 1) * 5) * cnt;
                    } else {
                        f[i] = 0;
                        t -= ((i + 1) * 5) * f[i];
                    }
                }
            }
            if (t != 0) return false;
            f[(bill/5) - 1] ++; 
        }
        return true;
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

## 第 108 场双周-2023/7/8

### [灵神的题解](https://www.bilibili.com/video/BV1Nh4y1E7cP)

### [2765. 最长交替子序列](https://leetcode.cn/problems/longest-alternating-subarray/)

```c++
class Solution {
public:
    int alternatingSubarray(vector<int>& nums) {
        int n = nums.size();
        int flag = 1, res = 0, cnt = 0;
        // 预处理差
        for (int i = n - 1; i >= 1; -- i) {
            nums[i] -= nums[i - 1];
        }
        for (int i = 1; i < n; ++ i) {
            if (nums[i] == flag) {
                ++cnt;
                flag = -flag;
            } else {
                res = max(res, cnt);
                flag = 1;
                cnt = 0;
                if (nums[i] == 1) --i;
            }
        }
        res = max(res, cnt);
        return res == 0 ? -1 : res + 1;
    }
};
```

### [2766. 重新放置石块](https://leetcode.cn/problems/relocate-marbles/)

```c++
class Solution {
public:
    vector<int> relocateMarbles(vector<int>& nums, vector<int>& moveFrom, vector<int>& moveTo) {
        vector<int> res;
        map<int,int> cnt;
        int n = moveFrom.size();
        for (auto& x : nums) {
            cnt[x]++;
        }
        for (int i = 0; i < n; ++ i) {
            int from = moveFrom[i], to = moveTo[i];
            if (cnt[from]) {
                cnt[from] = 0;
            }
            cnt[to] = 1;
        }
        for (auto it : cnt) {
            if (it.second > 0) {
                res.push_back(it.first);
            }
        }
        return res;
    }
};
```

### [2767. 将字符串分割为最少的美丽子字符串](https://leetcode.cn/problems/partition-string-into-minimum-beautiful-substrings/)

#### dfs回溯

```c++
class Solution {
public:
    int INF = 0x3f3f3f3f;
    string tenTobit(int n) {
        string res = "";
        while (n > 0) {
            if (n % 2 == 1) res += "1";
            else res += "0";
            n /= 2;
        }
        reverse(res.begin(), res.end());
        return res;
    }
    int minimumBeautifulSubstrings(string s) {
        vector<string> str5;
        int len = s.size();
        int num = 1;
        // 先预处理得到长度为len及一下的5的幂次的二进制
        while (num < (1 << len + 1)) {
            // 化成二进制
            str5.push_back(tenTobit(num));
            num *= 5;
        }
        // 深搜
        function<int(int)> dfs = [&](int i){
            if (i == len) return 0;
            // 美丽串不能前导零
            if (s[i] == '0') {
                return INF;
            } 
            int res = INF;
            for (string str : str5) {
                if (i + str.size() > len) break;
                // bug: substr(begin, size());
                string x = s.substr(i, str.size());
                if (x.compare(str) == 0) {
                    // 利用回溯来计算
                    res = min(res, dfs(i + str.size()) + 1);
                }
            }
            return res;
        };
        auto ans = dfs(0);
        return ans == INF ? -1 : ans;
    }
};
```

#### 把dfs递推成dp

```c++
class Solution {
public:
    int INF = 0x3f3f3f3f;
    string tenTobit(int n) {
        string res = "";
        while (n > 0) {
            if (n % 2 == 1) res += "1";
            else res += "0";
            n /= 2;
        }
        reverse(res.begin(), res.end());
        return res;
    }
    int minimumBeautifulSubstrings(string s) {
        vector<string> str5;
        int len = s.size();
        int num = 1;
        int f[len + 1];
        while (num < (1 << len + 1)) {
            // 化成二进制
            str5.push_back(tenTobit(num));
            num *= 5;
        }
        memset(f, INF, sizeof f);
        f[len] = 0;
        for (int i = len - 1; i >= 0; -- i) {
            // 有前导零就会直接continue过去，直接会INF
            if (s[i] == '0') continue;
            for (string str : str5) {
                if (i + str.size() > len) break;
                if (str.compare(s.substr(i, str.size())) == 0) {
                    f[i] = min(f[i], f[i + str.size()] + 1);
                }
            }
        }
        
        return f[0] == INF ? -1 : f[0];
    }
};
```



### [2768. 黑格子的数目](https://leetcode.cn/problems/number-of-black-blocks/)

```c++
class Solution {
public:
    vector<long long> countBlackBlocks(int m, int n, vector<vector<int>>& coordinates) {
        vector<long long> cnt(5, 0);
        set<pair<int,int>> s;
        set<pair<int,int>> vis;
        // 加入
        for (vector<int> p : coordinates) {
            s.insert({p[0], p[1]});
        }
        for (auto [x, y] : s) {
            // 枚举右上角
            for (int i = max(1, x); i < min(m, x + 2); ++ i) {
                for (int j = max(1, y); j < min(n, y + 2); ++ j) {
                    if (!vis.count({i, j})) {
                        vis.insert({i, j});
                        // 计算以该点为左上角的块的黑点数量
                        int sum = (s.count({i,j}) != 0) + (s.count({i - 1, j}) != 0) + (s.count({i,j - 1}) != 0) + (s.count({i - 1, j - 1}) != 0);
                        cnt[sum]++;
                    }
                }
            }
        }
        // for (int i = 0; i < 5; ++ i) cout << cnt[i] << " ";

        // cout << vis.size() << "\n";
        cnt[0] = (long long)(m - 1) * (n - 1) - vis.size();
        return cnt;
    }
};
```

## 第 353 场周赛-2023/7/9

### [灵神题解](https://www.bilibili.com/video/BV1XW4y1f7Wv/)

### [2769. 找出最大的可达成数字](https://leetcode.cn/problems/find-the-maximum-achievable-number/)

```c++
class Solution {
public:
    int theMaximumAchievableX(int num, int t) {
        return num + t * 2;
    }
};
```

### [2770. 达到末尾下标所需的最大跳跃次数](https://leetcode.cn/problems/maximum-number-of-jumps-to-reach-the-last-index/description/)

#### dfs + 记忆化做法

```c++
class Solution {
public:
    int maximumJumps(vector<int>& nums, int target) {
        int n = nums.size();
        int f[n];
        for (int i = 0 ; i < n ;++ i) f[i] = INT_MIN;
        f[n - 1] = 0;
        function<int(int)> dfs = [&](int x){
            // 用记忆化减少次数
            if (f[x] != INT_MIN) return f[x];
            int res = INT_MIN;
            for (int i = x + 1; i < n; ++ i) {
                if (abs(nums[i] - nums[x]) <= target) {
                    res = max(res, dfs(i) + 1);
                }
            } 
            f[x] = res;
            return f[x];
        };
        int ans = dfs(0);
        return ans < 0 ? -1 : ans;
    }
};
```



#### dp做法

```c++
class Solution {
public:
    int maximumJumps(vector<int>& nums, int target) {
        int n = nums.size(), INF = -1;
        int f[n + 1];
        memset(f, INF, sizeof f);
        f[0] = 0;
        for (int i = 1; i < n; ++ i) {
            for (int j = 0; j < i; ++ j) {
                if (f[j] != -1 && abs(nums[i] - nums[j]) <= target) {
                    f[i] = max(f[i], f[j] + 1);
                }
            }
        }
        return f[n - 1];
    }
};
```

### [2771. 构造最长非递减子数组](https://leetcode.cn/problems/longest-non-decreasing-subarray-from-two-arrays/description/)

#### dfs + 记忆化做法

```c++
class Solution {
public:
    int maxNonDecreasingLength(vector<int>& nums1, vector<int>& nums2) {
        int n = nums1.size();
        int nums[n][2];
        // 记忆化
        int vis[n][2];
        memset(vis, -1, sizeof vis);
        for (int i = 0; i < n; ++ i) {
            nums[i][0] = nums1[i];
            nums[i][1] = nums2[i];
        }
        // dfs(i,j)是以结尾nums[i]结尾的，选第j个的情况
        function<int(int,int)> dfs = [&](int i, int j)->int{
            if (i == 0) return 1;
            // 使用记忆化减少次数
            if (vis[i][j] != -1) return vis[i][j];
            int res = 1;
            if (nums1[i - 1] <= nums[i][j]) 
                res = dfs(i - 1, 0) + 1;
            if (nums2[i - 1] <= nums[i][j]) 
                res = max(res, dfs(i - 1, 1) + 1);
            vis[i][j] = res;
            return res;
        };
        int ans = 0;
        for (int i = 0; i < n; ++ i) {
            ans = max(ans, dfs(i, 0));
            ans = max(ans, dfs(i, 1));
        }
        return ans;
    }
};
```

#### 化成dp递推

```c++
class Solution {
public:
    int maxNonDecreasingLength(vector<int>& nums1, vector<int>& nums2) {
        int n = nums1.size(), res = 1;
        int nums[n][2], f[n][2];
        for (int i = 0; i < n; ++ i) {
            nums[i][0] = nums1[i];
            nums[i][1] = nums2[i];
        }
        for (int i = 0; i < n; ++ i) {
            f[i][0] = f[i][1] = 1;
        }
        f[0][0] = f[0][1] = 1;
        for (int i = 1; i < n; ++ i) {
            for (int j = 0; j < 2; ++ j) {
                if (nums1[i - 1] <= nums[i][j]) {
                    f[i][j] = f[i - 1][0] + 1;
                } 
                if (nums2[i - 1] <= nums[i][j]) {
                    f[i][j] = max(f[i][j], f[i - 1][1] + 1);
                }
                res = max(res, f[i][j]);
            } 
        }
        return res;
    }
};
```

### [2772. 使数组中的所有元素都等于零](https://leetcode.cn/problems/apply-operations-to-make-all-array-elements-equal-to-zero/description/)

维护差分数组，求差分数组的合理性

```c++
class Solution {
public:
    bool checkArray(vector<int>& nums, int k) {
        int n = nums.size();
        vector<int> d(n + 1);
        d[0] = nums[0];
        d[n] = -nums[n - 1];
        for (int i = n - 1; i >= 1; -- i) d[i] = nums[i] - nums[i - 1];
        for (int i = 0; i + k <= n; ++ i) {
            if (d[i] > 0) {
                d[i + k] += d[i];
                d[i] = 0;
            } else if (d[i] < 0) return false;
        }
        for (int i = 0; i <= n; ++ i) {
            if (d[i]) return false;
        }
        return true;
    }
};
```

## [第 354 场周赛 - 2023/7/16](https://leetcode.cn/contest/weekly-contest-354/)

### [6889. 特殊元素平方和](https://leetcode.cn/problems/sum-of-squares-of-special-elements/)

```c++
class Solution {
public:
    int sumOfSquares(vector<int>& nums) {
        int n = nums.size();
        int ans = 0;
        for (int i = 0; i < n; ++ i) 
            if (n % (i + 1) == 0) {
                ans += nums[i] * nums[i];
            }
        return ans;
    }
};
```

### [6929. 数组的最大美丽值](https://leetcode.cn/problems/maximum-beauty-of-an-array-after-applying-operation/description/)

```c++
class Solution {
public:
    int maximumBeauty(vector<int>& nums, int k) {
        int n = nums.size();
        sort(nums.begin(), nums.end());
        int ans = 0;
        // 1, 2, 4, 6
        for (int i = 0; i < n; ++ i) {
            int l = nums[i];
            int r = nums[i] + 2 * k;
            int lindex = lower_bound(nums.begin(), nums.end(), l) - nums.begin();
            int rindex = upper_bound(nums.begin(), nums.end(), r) - nums.begin();
            ans = max(ans, rindex - lindex);
        }
        return ans;
    }
};
```

### [6927. 合法分割的最小下标](https://leetcode.cn/problems/minimum-index-of-a-valid-split/)

```c++
class Solution {
public:
    int minimumIndex(vector<int>& nums) {
        int n = nums.size();
        int x = 0, freq = 0;
        map<int,int> cnt;
        // 求支配数
        for (int i = 0; i < n; ++ i) {
            cnt[nums[i]]++;
            if (cnt[nums[i]] > freq) {
                freq = cnt[nums[i]];
                x = nums[i];
            }
        }
        cout << x << '\n';
        int freq_l = 0, freq_r = freq - freq_l;
        for (int i = 0; i < n; ++ i) {
            int len_l = i + 1, len_r = n - 1 - i;
            if (nums[i] == x) {
                freq_l++;
                freq_r--;
            }
            cout << freq_l << " " << len_l << " " << freq_r << " " << len_r <<'\n';
            if (freq_l > len_l / 2 && freq_r > len_r / 2) return i;
        }
        
        return -1;
    }
};
```

### [6924. 最长合法子字符串的长度](https://leetcode.cn/problems/length-of-the-longest-valid-substring/description/)

双指针

```c++
class Solution {
public:
    int longestValidSubstring(string word, vector<string>& forbidden) {
        unordered_set<string> set{forbidden.begin(), forbidden.end()};
        int ans = 0, left = 0, n = word.size();
        for (int right = 0; right < n; ++ right) {
            // 倒着找就是最长的
            for (int i = right; i >= left && i > right - 10; -- i) {
                if (set.count(word.substr(i, right - i + 1))) {
                    left = i + 1;
                    break;
                }
            }
            ans = max(ans, right - left + 1);
        }
        return ans;
    }
};
```

