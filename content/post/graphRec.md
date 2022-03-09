---
title: "GraphRec : Graph Neural Networks for Social Recommendation 리뷰"
date: 2022-03-09T20:45:24+09:00
draft: false
categories: ["Recommendation"]
tags: ["Recsys", "GNN", "GraphRec", "JD"]

---

# ABSTRACT

소셜 추천 시스템의 데이터는 아래와 같은 표현 된다.

- 유저 - 유저 소셜 그래프
- 유저 - 아이템 그래프

그리고 핵심은 유저와 아이템의 Latent Factor로 학습하는 것이다.

그러나, GNNs으로 소셜 추천 시스템을 구축하는 것은 세가지 과제 있다.

1. 유저-아이템 그래프가 상호작용과 그에 연관된 오피니언을 포함한다
2. 소셜 관계들이 서로 다른 강도를 가진다
3. 유저가 유저-유저 소셜 그래프와 유저-아이템 그래프 모두에 포함된다

위의 세가지 문제를 해결하기 위해 논문이 기여하는 바는 아래와 같다.

- 유저-아이템 그래프의 상호작용과 오피니언을 동시에 포착하는 방법
- 두 그래프와 서로 다른 강도를 일관되게 모델링하는 GraphRec 프레임워크
- 두 실제 데이터셋 기반의 GraphRec의 효과성 증명

# 1. INTRODUCTION

소셜 추천 시스템은 아래와 같은 실제 현상에 기반해서 개발되고 있다.

> 유저는 보통 정보를 주변(친구, 동료)에서 획득한다.
> 

→ 소셜 관계가 추천 성능 증진에 도움이 된다가 증명되고 있다.

그래프 뉴럴 네트워크 ( GNN )의 아이디어

> 뉴럴 네트워크를 사용해서 그래프 이웃들로 부터 피처 정보를 어떻게 반복적으로 집계할 것인가?
> 

소셜 추천의 데이터는 두개의 그래프의 그래프 데이터로 표현될 수 있다. 아래의 그림 1에서 증명 되듯이 소셜 그래프와 유저-아이템 그래프를 가진다.

{{< figure src="/images/graphRec/0.png" title="Figure 1" >}}

유저 - 아이템 그래프 : 유저와 아이템 간의 상호작용을 나타낸다.

소셜 그래프( 유저 - 유저 그래프 ) : 유저 들 간의 관계를 나타낸다.

여기서 유저는 두 그래프 모두에 포함되어 두 그래프를 연결하는 브릿지가 된다.

소셜 추천 시스템 구축의 핵심은 유저 표현과 아이템 표현을 학습하는 것이다.

그러나, GNN으로 소셜 추천 시스템을 구축하는 것에 고려해야할 점이 있다.

1. 두 그래프를 어떻게 일관되게 결합할 것인가?
    
    두 그래프가 유저에 대한 정보를 제공하지만, 서로 다른 측면(소셜, 아이템)의 정보를 제공한다. 더 나은 유저 표현을 학습하기 위해 정보를 집계하는 과정이 중요하다.
    
2. 유저와 아이템 간의 상호작용과 오피니언을 어떻게 함께 포착할 것인가?
    
    유저-아이템 그래프가 유저와 아이템 간의 상호작용 뿐만이 아닌 아이템에 대한 오피니언도 포함한다. 
    
    예 ) 바지와 노트북에 대한 상호작용은 하지만, 바지는 좋고! 노트북은 싫어!
    
3. 서로 다른 강도를 가진 소셜 관계를 어떻게 구분할 것인가?
    
    유저는 강한 유대 관계를 가진 상대와 유사한 취향을 더 많이 공유한다. 그런데 약한 유대관계도 함께 섞여 였다. 소셜 관계를 동일하게 고려하는 것이 추천 성능을 감소시킨다.
    

이 연구에서는, 언급된 과제들을 동시에 해결하는 GraphRec을 제시한다.

- 소셜 추천의 그래프 데이터를 일관성 있게 모델링하는 GraphRec 제시
- 유저-아이템 그래프의 상호작용과 오피니언을 함께 포착하는 원칙적 접근법 제시
- 소셜 관계의 서로 다른 강도를 수학적으로 고려하는 메소드 소개
- 다양한 실제 데이터셋으로 제시된 프레임워크의 효과성 증명

# 2. THE PROPOSED FRAMEWORK

## 2.1 Definitions and Notations

유저-아이템 그래프 $R$과 소셜 그래프 $T$가 주어 졌을 때, 유저-아이템 그래프 $R$의 결측된 레이팅 값을 예측하는 것이 목표이다. 

- 유저 셋 : $U = {\{ u_1, u_2, ..., u_n }\}$, $n$명의 유저
- 아이템 셋 : $V = {\{ v_1, v_2, ..., v_m }\}$, $m$개의 아이템
- 유저 - 아이템 그래프 : $R \in R^{n \times m}$ ( $r_{i, j } = 0 \ or \ 1$ )
- 유저 - 유저 소셜 그래프 :  $T \in R^{n \times n }$ ( $T_{ij} = 0 \ or  \ 1$ )

{{< figure src="/images/graphRec/1.png" title="Table 1" >}}

## 2.2 An Overview of the Proposed Framework

모델의 아키텍처는 그림 2와 같다.

{{< figure src="/images/graphRec/2.png" title="Figure 2" >}}

모델은 세가지 컴포넌트로 구성된다.

1. 유저 모델링
    
    소셜 그래프와 유저-아이템 그래프 두 그래프의 서로 다른 측면을 학습한다. 
    
    아이템 집계로 아이템 공간에서의 유저 표현을 학습하고 소셜 집계로 소셜 공간에서의 유저 표현을 학습 한 뒤, 결합해서 User Latent Factor를 획득한다.
    
2. 아이템 모델링
    
    유저 집계로 유저-아이템 그래프의 상호작용과 오피니언을 고려해서 유저의 오피니언으로 Item Latent Factor를 획득한다.
    
3. 레이팅 예측
    
    유저와 아이템 모델링 컴포넌트를 통합해서 예측값을 생성하는 모델 파라미터를 학습한다.
    

## 2.3 User Modeling

유저 모델링의 목표는 User Latent Factors를 학습하는 것이다. 

유저 $i$를 $d$차원의 벡터로 표현한다. ( $h_i \in R^d$ ) 

어떻게 유저-아이템 그래프와 소셜 그래프를 결합하는 방법을 다룬다.

1. 첫번째 집계 : 유저-아이템 그래프에서 item-space user latent factor (  $h_i^I \in R^d$ )
2. 두번째 집계 : 소셜 그래프에서 social-space user latent factor ( $h_i^S \in R^d$ )
3. 두 집계를 결합해서 최종 user latent factor를 생성한다. ( $h_i$ )

### Item Aggregation ( $h_i^I$ )

유저-아이템 그래프는 유저와 아이템 간 상호작용과 아이템에 대한 유저의 오피니언을 포함한다.

유저 $u_i$가 상호작용한 아이템 들과 이 아이템 들에 대한 유저의 오피니언들을 고려한다.

{{< figure src="/images/graphRec/3.png" title="" >}}

- $C(i)$ - 유저 $u_i$가 상호작용한 아이템 셋 ( 유저-아이템 그래프에서의 $u_i$의 이웃들 )
- $x_{ia}$- 유저 $u_i$와 아이템 $v_a$ 간 오피니언( $r$, 레이팅 점수 )을 인지한 상호작용
- $Aggre_{items}$ - 집계 함수
- $\sigma$ - RELU

유저는 유저-아이템 상호작용 중 아이템에 대한 그/그녀의 의견을 $r$로 표현할 수 있다. 아이템에 대한 이러한 의견들로 아이템에 대한 유저의 선호도로 포착해서 item-space user latent factor를 모델링 하는데 도움을 줄 수 있다.

{{< figure src="/images/graphRec/4.png" title="" >}}

- $q_a$ - 아이템 임베딩
- $e_r$ - 오피니언 임베딩
- $g_v$- MLP

각 오피니언 $r$을 Dense Vector 표현으로 오피니언 임베딩 벡터 $e_r \in R^d$로 사용한다. ( {1,2,3,4,5} → $e_r$ )

아이템 임베딩 $q_a$과 오피니언 임베딩 $e_r$을 concatenate 한 후 MLP에 입력한 출력 결과 $x_{ia}$

상호 작용한 아이템 셋의 $Aggre_{items}$ 집계 함수에 사용할만한 유명한 연산은 평균이다. 

{{< figure src="/images/graphRec/5.png" title="" >}}

- $\alpha_i$-모든 아이템에 대한  $\frac{1}{|C(i)|}$로 고정한다.

유저 $u_i$를 표현을 생성하는데 모든 상호 작용이 동등하게 기여한다고 가정한다.

그러나 유저를 표현하기 위한 상호작용 들의 영향도는 극적으로 달라질 것이다. 그래서 각각의 상호작용에 가중치를 부여해서 상호작용 들이 서로 다르게 기여하게 만들어야 한다. 따라서 어텐션 메커니즘을 적용한다.

{{< figure src="/images/graphRec/6.png" title="" >}}

- $\alpha_{ia}$ - $C(i)$에 포함된 각각 아이템 $v_a$에 대한 유저 $u_i$의 상호 작용 $( v_a, u_i )$ 어텐션 가중치

아이템 어텐션 $\alpha_{ia}$를 어텐션 네트워크라 부르는 두 레이어의 뉴럴 네트워크로 구성한다. 어텐션 네트워크는 아래와 같이 정의한다.

{{< figure src="/images/graphRec/7.png" title="" >}}

- $x_{ia}$ - 오피니언을 인식한 상호작용
- $p_i$ - 타겟 유저 $u_i$의 임베딩
- $w_2^T$  : Query
- $x_{ia}$ : Value

최종적인 어텐션 가중치는 위에서 계산한 어텐션 스코어를 소프트맥스 함수를 사용해서 도출한다.

{{< figure src="/images/graphRec/8.png" title="" >}}

### Social Aggregation ( $h_i^S$ )

소셜 상관 계수 이론에 따르면, 유저의 선호도는 그/그녀와 직접적으로 연결된 소셜 친구와 유사하거나 혹은 영향을 받는다. 따라서 user latent factor를 추가로 모델링하기 위해 소셜 정보를 통합 시켜야 한다. 그러나 서로 다른 소셜 관계 강도를 고려해서 social-space user latent factors를 학습해야 한다. 따라서 소셜 친구를 선택하는 과정에 어텐션 메커니즘을 사용해서 유저 소셜 정보를 강조한 다음, 그 정보를 집계해야 한다.

{{< figure src="/images/graphRec/9.png" title="" >}}

- $N(i)$ - 유저 $u_i$의 이웃 셋

유저의 $u_i$의 이웃 셋의 item-space user latent factors를 집계해서 social-space user latent factor, $h_i^S$를 구성한다.

{{< figure src="/images/graphRec/10.png" title="" >}}

마찬가지로, 평균을 사용한 집계가 가장 자연스러운 방법이지만, 강한 유대와 약한 유대가 소셜 네트워크에 공존하고, 강한 유대를 가진 유저들 간에 더 많은 취향을 공유하므로 이러한 유저들을 뽑는데 레이어가 두개인 어텐션 메커니즘을 수행한다. 

{{< figure src="/images/graphRec/11.png" title="" >}}

- $\beta_{io}$- 유저 들 간의 유대 강도

### Learning User Latent Factor

더 좋은 user latent factors를 학습 하기 위해,  item-space user latent factor와 social-space user latent factor가 함께 고려되어야 한다. 따라서 두 latent factor를 concatenate 한 후, 표준 MLP 통과 시켜 최종 user latent factor를 생성한다. 

{{< figure src="/images/graphRec/12.png" title="" >}}

- $l$ - 히든 레이어의 인덱스

## 2.4 Item Modeling

아이템 $v_j$에 대해 유저 집계로 item latent factor, $z_j$를 학습한다. 아이템들은 유저-아이템 그래프와 연관되는데, 상호작용과 유저의 오피니언을 포함한다. 따라서, 유저-아이템 그래프의 오피니언과 상호작용이 동시에 포착되어서 더 나은 item latent factor를 학습해야 한다.

### User Aggregation

아이템 집계를 통한 item-space user latent factor를 학습한 메소드와 유사한 메소드를 사용한다. 각각의 아이템 $v_j$에 대해, $v_j$와 상호작용한 유저 셋 $B(j)$의 정보를 집계해야 한다.

동일한 아이템이라도, 유저들이 유저-아이템 상호작용 중에 다른 오피니언을 표현할 수도 있다. 이런 서로 다른 유저들의 오피니언 들이 동일한 아이템의 특징을 포착해서, item latent factor를 모델링 하는데 도움을 준다. 

{{< figure src="/images/graphRec/13.png" title="" >}}

기본 유저 임베딩과 오피니언 임베딩을 MLP에 통과시켜 오피니언을 인식한 유저 벡터를 얻는다.

{{< figure src="/images/graphRec/14.png" title="" >}}

그리고 오피니언을 인식한 유저 벡터를 집계한다.

{{< figure src="/images/graphRec/15.png" title="" >}}

어텐션 메커니즘을 적용해서 유저들의 중요도 가중치에 차별을 둔다. 

## 2.5 Rating Prediction

아이템 랭킹과 레이팅 예측 같은 다양한 추천 태스크가 있다. 이 연구에서는 레이팅 예측을 적용했다. 유저들과 아이템들의 latent factor ( $h_i$ 와 $z_j$)를 concatenate 한 후, 레이팅 예측을 위해 MLP 레이어에 통과시킨다.

{{< figure src="/images/graphRec/16.png" title="" >}}

## 2.6 Model Training

이 연구의 태스크가 레이팅 예측이기 때문에, 공통적으로 사용되는 목적 함수는 아래와 같다. 

{{< figure src="/images/graphRec/17.png" title="" >}}

- $|O|$- 관측된 레이팅 수
- $r_{ij}$ - 그라운드 트루스 레이팅

목적함수 최적화를 위해 RMSprop을 사용했다. 세개의 임베딩은 원핫벡터를 사용하지 않고 랜덤하게 초기화 해서 사용했다. 오버피팅 방지를 위해 dropout도 적용했다. 예측에는 전체 네트워크가 사용된다.

# 3. EXPERIMENT

## 3.1 Experimental Settings

### 3.1.1. Datasets

Ciao와 Epinions 의 두 데이터셋을 사용했다. 근데 저자가 [JD.com](http://JD.com) 인데 왜?

- Ciao
    
    Ciao는 영국, 프랑스, 스페인, 독일, 네덜란드, 이탈리아 및 스웨덴에 웹 사이트가있는 유럽 기반 온라인 쇼핑 포털입니다
    
- Epinions
    
    Epinions.com은 1999 년에 설립 된 일반 소비자 검토 사이트입니다. Epinions는 2003 년 DealTime에 의해 인수되었으며 이후 Shopping.com은 2005 년에 eBay에 의해 인수되었습니다.
    
- JD
    
    징동상청은 중화인민공화국에서 360buy를 경영하고 있던 류창동이 설립한 웹서비스 회사이다. 중국 베이징시 차오양 구에 본사를 세워, 같은 사의 EC사이트인 "[JD.com](http://jd.com/)"에서는 가전・PC・가구・의류・식품・도서 등의 상품을 인터넷으로 판매하고 있다.
    

{{< figure src="/images/graphRec/18.png" title="Table 2" >}}

### 3.1.2 Evaluation Metrics

RMSE와 MAE로 예측 정확도 검증 했다. 값이 작을 수록 좋다.

### 3.1.3

기존 추천 시스템, 뉴럴 넷 기반 추천 시스템 그리고 기존 소셜 추천 시스템을 세 그룹의 메소드를 비교했다. 추가로 소셜 추천 시스템 기반의 SOTA 뉴럴 네트워크 DeepSoR 과 GCMC+SN을 GraphRec과 비교했다.

- PMF, NeuMF
    
    소셜 네트워크 정보가 없는 순수한 CF 모델
    
- SoRec, SoReg, SocialMF, TrustMF, NeuMF, DeepSoR, GCMC+SN
    
    소셜 추천 모델
    

## 3.2 Performance Comparision of Recommender Systems

{{< figure src="/images/graphRec/19.png" title="Table 3" >}}

- SoRec, SoReg, SocialMF, TrustMF가 항상 PMF에 비해 좋은 성능을 보인다. PMF는 레이팅 정보만 사용하지만, 나머지 모델은 추가로 소셜 네트워크 정보를 활용한다. 소셜 네트워크 정보를 사용하는 것이 좋아 보인다.
- 뉴럴 네트워크를 사용한 NeuMF가 PMF에 비해 더 좋은 성능을 보인다. 그리고 모두 소셜 네트워크 정보를 추가로 사용하지만 DeepSoR, GCMC+SN이 SoRec, SoReg, SocialMF, TrustMF에 비해 성능이 더 좋다. 뉴럴 네트워크 모델을 사용하는 것이 좋다.
- GCMC+SN이 베이스라인들에 비해 성능이 좋다. GNN을 사용하는 것이 좋다.
- GraphRec의 성능이 가장 좋다.

## 3.3 Model Analysis

### 3.3.1. Effect of Social Network and User Opinions

GraphRec 동작을 확인하기 위해 변형한 GraphRec-SN과 GraphRec-Opinion을 GraphRec을 비교한다.

- GraphRec-SN : GraphRec에서 소셜 네트워크 정보를 제거한 모델
- GraphRec-Opinion : GraphRec에서 오피니언 임베딩을 제거한다.

{{< figure src="/images/graphRec/20.png" title="Figure 3" >}}

GraphRec-SN의 성능이 GraphRec에 비해 나쁘다. → 소셜 네트워크 정보가 성능에 중요하다.

GraphRec-Opinion의 성능이 GraphRec에 비해 나쁘다. → 유저-아이템 상호작용의 오피니언 정보가 중요하다.

### 3.3.2. Effect of Attention Mechanism

GraphRec의 키 컴포넌트인 집계에 사용한 어텐션 메커니즘을 검증한다. 아이템 어텐션, 소셜 어텐션, 유저 어텐션을 비교한다.

- GraphRec-$\alpha$ : 아이템 집계에 사용된 아이템 어텐션을 제거하고 Mean 집계를 사용한다.
- GraphRec-$\beta$ : 소셜 집계에 사용된 소셜 어텐션을 제거하고 Mean 집계를 사용한다.
- GraphRec-$\alpha$&$\beta$ : 아이템 어텐션과 소셜 어텐션을 제거하고 Mean 집계를 사용한다.
- GraphRec-$\mu$ : 유저 집계에 사용한 유저 어텐션을 제거하고 Mean 집계를 사용한다

{{< figure src="/images/graphRec/21.png" title="Figure 4" >}}

 한 유저가 상호작용한 아이템 들이 item-space user latent factor에 동일하게 기여하지 않았다. 그리고 item latent factor를 학습하는데 상호작용한 모든 유저들이 동일한 중요성으로 기여하지 않는다. 라는 가정에서 유저들과 아이템들에 대해 어텐션 메커니즘을 적용한다. GraphRec-$\alpha$과 GraphRec-$\mu$의 성능이 GraphRec에 비해 나쁜데 아이템 집계와 유저 집계에 어텐션을 사용하는데 장점이 있다는 것을 증명한다.

유저들이 강한 유대감을 가진 상대와 비슷한 취향을 더 많이 공유한다고 한다. GraphRec-$\beta$의 성능이 떨어지는데, 이는 소셜 관계의 중요도를 구분하는 것이 중요하다는 것을 보인다.

### Effect of Embedding Size

유저 임베딩 $p$, 아이템 임베딩 $q$, 그리고 오피니언 임베딩 $e$의 사이즈를 분석한다. 

{{< figure src="/images/graphRec/22.png" title="Figure 5" >}}

임베딩 사이즈가 증가할 수록 성능이 증가하다가 다시 떨어지는데, 임베딩 사이즈가 커질 수록 강력한 표현력을 가지지만, 모델의 복잡도도 증가하는 것을 증명한다. 

# 5. CONCLUSION AND FUTURE WORK

레이팅 예측을 위한 소셜 추천을 모델링하기 위한 그래프 네트워크 모델인 GraphRec을 제시했다.

유저-아이템 그래프에서 상호작용과 오피니언을 동시에 포착하기 위한 방법을 제시했다.

추가로, GraphRec이 소셜 관계의 강도를 구분할 수 있다. 실제 데이터 셋으로 성능을 증명했다.

현재는 레이팅과 소셜 통계 정보를 함께 고려하고 있는데, 정보들이 다이나믹하다. 따라서 추후 연구에서는 소셜 추천을 위한 다이나믹 그래프 뉴럴 네트워크를 구축을 고려할 것이다.