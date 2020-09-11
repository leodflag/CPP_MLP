# CPP_MLP
Multilayer perceptron

## 目標
    使用多層感知機(Multilayer perceptron, MLP)解決AND、OR、XOR問題
## 參數設定
    誤差值：0.05
    初始權重與修正權重矩陣：神經元數量：3，權重+閾值：3

    AND
    {{-1,-1,1},
    {-1,1,0},
    {1,-1,0},
    {1,1,1}}
    學習速率：10.0

    OR
    {{-1, -1, 0}, 
    {-1, 1, 1}, 
    {1, -1, 1}, 
    {1, 1, 1}}
    學習速率：2.0

    XOR
    {{-1, -1, 0}, 
    {-1, 1, 1}, 
    {1, -1, 1}, 
    {1, 1, 0}}
    學習速率：2.0

## 虛擬碼
    1. 建立資料、初始權重與修正權重矩陣
    2. 建立隱藏層，存入初始權重矩陣
    3. 前向傳播
    4. 計算誤差
    5. 倒傳遞誤差
    6. 修正權重存入修正權重矩陣(BGD)
    7. 重複2.到6.，直到所有資料皆輸入NLP運算過
    8. 更新權重與閾值：將修正權重矩陣加至初始權重矩陣
    9. 重複7.到8.直到樣本的所有誤差相加小於誤差值


## 結果
AND
![image](https://github.com/leodflag/CPP_MLP/blob/master/AND_result.png)　　
OR
![image](https://github.com/leodflag/CPP_MLP/blob/master/OR_result.png)　　
XOR
![image](https://github.com/leodflag/CPP_MLP/blob/master/XOR_result.png)
