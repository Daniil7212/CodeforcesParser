import pyperclip


a = """#include <iostream>
#include <set>
#include <algorithm>
#include <map>
#include <vector>
#include <sstream>
#include <string>
#include <cmath> 
#include <iomanip>
#include <stack>
using namespace std;


int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string key,cmd,max1,t;
    //map<string, int> m;
    long long b=0,n,g,x1,x2,y1,y2,k, p=1,w,l=1,m;
    int s=-1000000000;
    cin >>  n >> m;

    vector<vector<int>> matrix(n,vector<int>(m));

    vector<vector<int>> dp(n+1,vector<int>(m+1,s));

    vector<string> hod;

    dp[0][1]=0;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j){
            cin >> matrix[i][j];
        }
    }
    
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j){
            dp[i][j] = max(dp[i-1][j], dp[i][j-1]) + matrix[i - 1][j -1];
        }
    }



    long long j=m,i=n, c=1;
    while(c<=n+m-2) {
        if (dp[i-1][j] > dp[i][j-1]){
            hod.push_back("D");
            i--;
        } else {
            hod.push_back("R");
            j--;
        }
        c++;
    }

    cout << dp[n][m] << endl;

    for (int j = hod.size()-1; j>=0; --j){
        cout << hod[j];
    }
}

"""

pyperclip.copy('\\n '.join(a.split('\n')))
