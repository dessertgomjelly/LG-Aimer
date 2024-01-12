---
layout: single
title: "[LG Aimers] 1. AI 윤리" 
categories: LG_Aimers
tag: [AI, 인공지능, AI윤리]
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



2024.01.02 ~ 2024.01.29 까지 한달 동안 온라인 AI 교육을 받고,  2024.02.01~2024.02.26까지 온라인 해커톤을 참여하는 여정입니다.



![module1]({{site.url}}/images/2023-01-08-LG Aimer Module 1/module1.png)

교육과정에 대하여 출처를 정확히 밝힌다면 블로그에 작성해도 된다는 답변을 받아 해당 과정을 블로그에 기록해보겠습니다.

<img src="{{site.url}}/images/2023-01-08-LG Aimer Module 1/module1 1.png" alt="module1 1" style="zoom:40%;" />

<img src="{{site.url}}/images/2023-01-08-LG Aimer Module 1/module1 2.png" alt="module1 2" style="zoom:50%;" />

<br>

<br>

# **1. AI 윤리**

-  교수
   -  KAIST 차미영 교수


-  학습 목표
   -  본 모듈은 본격적인 AI기술에 대한 이해에 앞서 데이터 과학자로서의 기본적 소양을 기르기 위한 과정입니다. 따라서 인공지능 기술 도입에 앞서 데이터 과학자로서 그리고 제도적으로 윤리적으로 어떠한 자세를 가져야 하는지 이해하고, 인공지능 기술로 어떻게 문제를 해결할 수 있는지 학습합니다.

<br>

<br>

## 데이터 전처리와 분석 방법은 적절한가?

- Error bar 추가하기
- 적합한 통계 테스트 찾기
- 아웃라이러 제거하기
- 데이터 표준화하기
- EDA (exploratory data analysis) 충분한 시간을 보내기

- 100만 데이터 건은 있어야지 많은 수의 파라미터를 학습 할 수 있다.
    - 언더 피팅
        - 모델이 너무 단순하다면 충분히 학습이 되지 않아서 적절한 선택하지 못한다.
    - 오버 피팅
        - 너무 과하게 학습한다면 데이터에 특화된 알고리즘이 나오기 때문에 데이터가 조금만 달라져도 쓸 수 없다.
- 데이터 학습의 결과가 적절한 수준인지에 대한 인식이 있어야한다.
- 학습(training)데이터는 테스트(testing) 데이터와 달라야한다.

<br>



## black box algorithnm

- 설명력이 중요한 AI 예시 : 탈세범 검출
- 실제 알고리즘에서 설명력을 높이기 위한 노력을 한다.
- 흔히 AI 기반 알고리즘은 설명 가능하지 않고 블랙바스 형태라는 단점 존재한다.
- High risk 결정에서는 설명력도 정확도 만큼이나 중요하다.
- Saliency map, SHAP과 같이 post-hoc explainability를 제공하는 기술이 생걌다.

<img src="{{site.url}}/images/2023-01-08-LG Aimer Module 1/module1 3-4707772.png" alt="module1 3" style="zoom:70%;" />

(post-hoc explainability)은 알고리즘의 내면을 가시화 해주는 사후 설명력이다.

- One pixel attack 의 예시에서는 픽셀 하나만 바뀔 경우 알고리즘 학습 결과가 달라진다.


<br>


## Handling the Web data

- 의견의 대표성 : Spiral of silence
    - 인터넷 상의 의견의 대표성 있는 의견이 아닐 수 있음을 인지해야 한다.
    - 편향 현상에 주의해야한다.
- 오정보의 빠른 확산으로 인한 인포데믹 현상
    - 인포데믹 : 사실정보와 더불어 오정보가 많아져 사실 구분이 힘든 현상이다.

<br>

<img src="{{site.url}}/images/2023-01-08-LG Aimer Module 1/module1 4.png" alt="module1 4" style="zoom:40%;" />

## 윤리

- GDPR : 과다 광고에 노출, 혹은 혐오 표현의 노출을 규제하는 법률이다.
- Digital Services Act
    - 유럽 연합 중심 빅테크 기업대상 플랫폼 유해 콘텐츠 단속 의무 강화해야한다.
    - 네티즌 성별,인종,종교 등에 기반한 알고리즘으로 개인화 추천 광고 노출 하지 않아야 한다.
    - 어린이 대상 개인화 추천 광고 금지한다.
    - 혐오 발언, 아동 학대, 테러 선동 등 불법 콘텐츠 유통 막아야한다.


<br>


## AI and Ethical Decisions

- 법 분야에서 COMPAS 제도
    - 피고의 미래 범죄 위험을 점수로 예측하는 Software Tool이다.
    - 판사가 결정을 내릴 때 참고하는 통계적 점수이다.
    - 편향 현상 : 인종차별 이슈
- Amazon AI기반 채용 시스템
    - 남성 지원자가 다수였던 과거 10년 이력서 데이터를 학습
    - 편향 현상 : 성별 이슈

<br>



## 데이터 분석과 AI 학습에서 유의 점

- 데이터의 확보, 전처리, 분석, 해석의 전 과정이 중요하다.!!!
- 알고리즘의 설명력, 편향, 신뢰의 문제에 주의해야 한다!


<br>


## AI Ethics

- 인간의 창조적 활동 영역으로 들어온 인공지능
    - AI가 기술혁신과 창작 도구로 활용이 점차 확대됨에 따라, 인간의 개입 없이 독자적 창작과 혁신활동이 가능한 수준으로 발전 하리라 전망한다.
- AI 시대 지식재산, 법인격, 처벌, 그리고 윤리의 문제 부각
    - AI에 의한 발명과 저작 등에 대한 법제 정비한다.
    - 미래 강한 AI가 등장했을 시 법인격을 부여할지 논의한다.
    - 오작동시 처벌과 윤리 규정을 마련한다.
    - 다양한 계층 시민의 수요와 요구를 반영하도록 유의한다.
