---
title: 5.1 개요
description: 5.1장은 프로젝트 전반을 소개한다. 파이썬과 MySQL로 주소 데이터베이스를 구축하고, 데이터 분석하는 방법을 학습한다.
keywords: [MySQL, Docker, Python, pymysql, 주소데이터분석]
url: "/chapter-5/chapter-5-1.html"
---

# 5.1 개요

5장의 프로젝트는 파이썬과 `MySQL`로 주소 데이터베이스를 구축하는 과정을 학습한다. 이 장의 목표는 파일 데이터를 넘어 대규모의 텍스트 데이터를 쉽게 저장하고 분석할 수 있는 `MySQL` 도구를 활용하는 데 있다.

## 학습 내용

이번 장은 파이썬 외에 `MySQL`이란 오픈소스 관계형 데이터베이스(relational database)를 사용한다.
데이터베이스를 구축하는 과정은 크게 4가지 단계로 구분한다.

- 1단계: 데이터베이스와 테이블 생성하기
- 2단계: 테이블 스키마 생성하기
- 3단계: 데이터 삽입∙삭제∙수정하기
- 4단계: SQL로 데이터 분석하기

기본적으로 `MySQL`은 데이터베이스와 그 안의 테이블로 구성된다. 활용 목적에 따라 여러 개의 데이터베이스를 생성할 수도 있고, 일반적으로 하나의 데이터베이스에 여러 개의 테이블이 존재한다. 테이블이 생성되면, 그 테이블에 데이터를 집어넣기 위해서 테이블의 스키마(schema)를 짜야한다. 테이블은 행과 열로 구성되고, 개별 열마다 어떤 컬럼명과 값이 들어갈 것인지를 선언해주는 것이다. 테이블의 스키마가 만들어진다면 데이터를 테이블에 삽입할 수 있다. 데이터의 삽입 뿐만 아니라 삭제, 수정 등의 작업도 수행할 수 있다. 데이터가 무사히 데이터베이스에 들어갔다면 `SQL`이란 별도의 질의 언어로 데이터를 탐색하거나 분석하는 것이 가능하다.

`SQL(Structured Query Language)`은 관계형 데이터베이스에 존재하는 데이터를 저장하거나 관리하기 위한 질의 언어다. 어떤 데이터베이스를 사용하냐에 따라 `SQL` 문법이 조금씩 달라질 수 있지만, 전반적으로 문법은 비슷하다. 이 장은 `MySQL`에서 `SQL`로 질의하는 방법에 대해 배우지만, 다른 데이터베이스(예: Oracle)와 문법이 비슷하기 때문에 전반적인 `SQL` 문법에 대해 학습할 수 있다.

## 학습 목표

이 장의 학습 목표는 다음과 같다.

1. 관계형 데이터베이스의 기본적인 사용방법을 익힌다.

관계형 데이터베이스는 대규모의 데이터 저장을 위해 광범위하게 사용되고 있다. 이 프로젝트는 `MySQL`을 사용해 관계형 데이터베이스의 기본적인 구조를 파악하고, 데이터를 업로드하고 분석할 수 있는 기본적인 방법을 제공한다.

2. 관계형 데이터베이스의 질의언어인 SQL의 간단한 구문을 배운다.

SQL은 관계형 데이터베이스를 다루는 데 필수적인 질의 언어다. `SELECT`, `CREATE`, `ALTER`, `UPDATE`와 같이 기본적인 SQL 구문을 익히고, SQL로 데이터베이스를 조작하는 방법을 배운다.

3. 파이썬에서 관계형 데이터베이스의 데이터를 가져와 분석한다.

앞 장에서 대부분의 데이터는 파일 데이터로 불러왔다. 이 장은 주소와 같은 대규모의 데이터를 데이터베이스에 저장하고 파이썬으로 데이터를 불러와 분석하는 방법을 학습한다.