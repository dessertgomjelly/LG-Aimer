---
layout: single
title: "[LG Aimers] 4-2. 지도학습(분류/회귀)"
categories: LG_Aimers
tag: [AI, 인공지능, Machine Learning, Linear Regression]
use_math: true #수학 공식 가능하게
sidebar:
    nav: "counts"


---

<style>
  body {
    font-size: 16px; /* 폰트 사이즈 조절 */
  }
</style>







**[공지사항]** [해당 내용은 LG에서 주관하는 LG Amiers : AI 전문가 과정 4기 교육 과정입니다.] 
[LG AI](https://www.lgaimers.ai/)
{: .notice--danger}

<br>
<br>





# **4-2. 지도학습(분류/회귀)**

-  교수 : 이화여자대학교 강제원 교수
-  학습목표
   -  Machine Learning의 한 부류인 지도학습(Supervised Learning)에 대한 기본 개념과 regression/classifiation의 목적 및 차이점에 대해 이해하고, 다양한 모델 및 방법론을 통해 언제 어떤 모델을 사용해야 하는지, 왜 사용하는지, 모델 성능을 향상시키는 방법을 학습하게 됩니다.



<br>

<br>

# *CHAPTER 4. Linear Classification*

<br>

<br>

## Linear Classification

-  Linear regression 모델과 같이 다음과 같은 수식을 따른다.

   <img src="/Users/dessert_gomjelly/Desktop/깃허브블로그/dessertgomjelly.github.io/images/2023-01-12-LG Aimer Module 4-2/image-20240112210948409.png" alt="image-20240112210948409" style="zoom:50%;" />

-  입력 freature *x* 와 그에 해당하는 파라미터셋 *w* 의 내적으로 구성되어있다.
   -  예측=*x*⋅*w*+*b*
-  가설 함수(hypothesis function)가 선형이라고 가정할 때, 이 함수의 결정 경계(decision boundary)는 데이터를 분리하는 초평면(hyperplane)이 된다.
   -  가설 함수 *h*(*x*)가 0보다 크면 양성 클래스(positive class), 0보다 작으면 음성 클래스(negative class)로 예측한다고 가정하면, 결정 경계는 가설 함수가 0이 되는 부분이다.
   -  결국 우리의 목적은 이러한 hyper plane을 구해서 우리의 data set에 있는 positive class 와 negetive class를 구분하는 것이다.

<img src="/Users/dessert_gomjelly/Desktop/깃허브블로그/dessertgomjelly.github.io/images/2023-01-12-LG Aimer Module 4-2/image-20240112211324966.png" alt="image-20240112211324966" style="zoom:50%;" />

-  Linear model 이 갖는 여러 장점 즉, 단순하며 해석 가능성이 있고 다양한 환경에서 일반적으로 안정적인 성능을 제공할 수가 있다.

