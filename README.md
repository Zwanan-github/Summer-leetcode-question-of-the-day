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

