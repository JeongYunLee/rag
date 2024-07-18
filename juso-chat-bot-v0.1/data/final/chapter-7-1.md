---
title: 7.1 프로젝트 소개
description: 7장에서는 데이터의 키 값으로 기능할 수 있는 도로명주소의 정제 방법과 데이터 품질평가 방법에 대해 안내한다.
keywords: [도로명주소, 데이터품질, 데이터정제, 공공데이터, geocoding, 네이버API]
url: "/chapter-7/chapter-7-1.html"
---
# 7.1 개요

## 데이터 품질의 중요성

데이터 분석 과정에서 하나의 데이터세트에 다른 데이터세트를 통합해야 하는 상황이 빈번히 발생한다. 

4장에서 다룬 지역별 인구 현황과 전국 공공시설 개방 현황 데이터세트를 결합하여 분석해야 한다고 가정해보자. 주민등록 인구 데이터세트와 공공시설 데이터세트를 어떤 기준으로 병합해야 할 것인가? 가장 적절한 방법은 시도, 시군구, 행정동 단위의 주소 정보를 활용하는 것이다. 공공데이터를 활용하다면, 서로 다른 데이터세트의 통합이 매우 중요하고,  주소 데이터는 데이터세트를 통합하는 데 효과적인 수단이 될 수 있다.

그러나 주소 데이터에 오류가 존재하거나, 데이터세트 통합 후에도 결측치가 과도하게 많거나, 데이터의 표기 방식이 불일치하는 경우가 발생한다면 어떠한 문제가 초래될 것인가? 이러한 경우 데이터의 활용에 앞서 정제 작업에 상당한 시간과 노력을 투자해야 하며, 심지어 정제 작업을 거쳤다 하더라도 데이터를 제대로 활용하기 어려울 수 있다.

따라서 다수의 데이터세트를 통합해야 하거나 데이터세트의 규모가 방대하다면, 본격적인 분석에 앞서 해당 데이터세트의 활용 가능성을 사전에 평가하는 과정이 필수적으로 요구된다. 이를 통해 불필요한 작업을 방지하고 분석의 효율성을 제고할 수 있다.

7장은 주소 데이터를 포함하는 데이터세트 전반의 품질을 분석하고, 주소 데이터가 주소구성요소에 부합하도록 올바르게 입력되었는지 평가하는 방법을 설명한다. 더불어 데이터세트 통합 시 주소 정제를 보다 효율적으로 수행할 수 있는 방안을 살펴본다. 

본 프로젝트는 샘플 데이터세트를 활용한 실습 위주로 진행되므로, 문서 내용과 함께 제시된 코드 예시를 참조하며 학습하는 것이 보다 용이할 것이다.




## 학습 내용

7장은 주소 데이터의 오류를 파악하고 정제하는 과정과 통합된 데이터셋의 품질을 평가하는 과정으로 구성된다.

1. 도로명주소의 오류 유형 알아보고 정제하기
     - 데이터를 합치기에 앞서 두 데이터에 존재하는 도로명주소 값의 오류를 파악하는 방법
     - 도로명주소의 오류를 정제한 다음 데이터를 합치기

2. 데이터의 품질 요소 알아보고 평가해보기
     - 데이터를 평가하는 품질 요소 알아보기
     - 합쳐진 데이터의 품질을 평가해보기

이 과정들에 대한 코드를 직접 실행하기 위해선, 자신만의 Naver Geocode API를 발급받아야 한다. 구글, 카카오에서도 Geocode API를 서비스하고 있으나, 네이버가 이들 중 가장 무료 요청 가능 건수가 많으므로 네이버를 선택하였다. 다른 API를 사용해도 무방하나, 이 경우 결과 값의 형식에 맞게 코드를 수정하는 과정이 필요하다. 다음 장에 이어서 네이버 GeocodeAPI 발급과 활용 방법에 대해 자세히 설명한다.