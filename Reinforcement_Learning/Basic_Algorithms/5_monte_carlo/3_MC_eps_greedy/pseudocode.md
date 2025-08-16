# $\epsilon$-greedy 算法伪代码

## 初始化:

for each $s \in S $, $a \in A$:
        $Q(s,a) \gets 0$        // 初始化动作价值估计
        $N(s,a) \gets 0$        // 初始化动作选择次数
set:  
$\epsilon \in (0,1)$​      // 设置初始探索概率   
$ episodes \in N $​​     //设置训练轮数     
$\gamma \in (0,1)$    //设置衰减率

## 算法

for $episode$ in range($episodes$):  
	update $\epsilon $   //更新探索率(一般是先大后小)  
	reset env, get initial state $s$    
	initialize data={}

​	while not (terminated or truncated):
​		Generate $r \sim Uniform(0,1)$  
​		if $r < \epsilon$:
​        	// 探索: 随机选择动作
​        		$a_t \gets random\ choice\ from\ A$
​    	    else:
​        	// 利用: 选择当前最优动作
​        		$a_t \gets \arg\max_{a} Q(s,a)$     
​		//执行动作进行交互  
​		env.step($a_t$),update state $s'$, observe reward $r$, record tuple($s,a,r$​) in data 

​	//更新 Q表  
​	$g\gets0$   
​	for $t$ in range(len(data)-1, -1, -1):    
​		$g\gets\gamma g+r_t$  
​		if $(s_t,a_t)$ is not visited:		
​			$N(s_t,a_t) \gets N(s_t,a_t) + 1$
​			$Q(s_t,a_t) \gets Q(s_t,a_t) + \frac{1}{N(s_t,a_t)} \left( g - Q(s_t,a_t) \right)$

