def bsp_value(a, n, m):
    l, r = 1, a[n - 1]
    ans = 0

    while l < r:
        mid = (l + r) // 2

        if valid(mid, a, n, m):
            ans = max(ans, mid)
            l = mid + 1
        else:
            r = mid

    return ans


def valid(mid, a, n, m):
    ix = a[0]
    c = 1
    elements = 0

    for i in range(1, n):
        if a[i] - ix >= mid:
            ix = a[i]
            c += 1
            if c == n-m:
                return True
    return False


def bsp_solution(mid, a, n, m):
    ix = a[0]
    c = 1
    sol_arr = []
    sol_arr.append(a[0])
    for i in range(1, n):
        if a[i] - ix >= mid:
            ix = a[i]
            sol_arr.append(a[i])
            c += 1
            if c == n - m:
                print(sol_arr)
                return True
    return False


# Example usage
a = [1, 2, 8, 9, 10, 18]  # [2, 4, 6, 7, 10, 14]
n = len(a)
m = 2

result = bsp_value(sorted(a), n, m)
print(result)
bsp_solution(result, a, n, m)  # ~O(nlog(n-m)+n)
