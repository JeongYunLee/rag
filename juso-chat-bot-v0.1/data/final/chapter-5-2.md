---
title: 5.2 프로젝트 환경 구축하기
description: 도커(Docker)를 사용해 MySQL을 설치하고, 파이썬 노트북으로 MySQL과 연결하는 방법에 대해 학습한다.
keywords: [MySQL, Docker, Python, pymysql, 주소데이터분석]
url: "/chapter-5/chapter-5-2.html"
---

# 5.2 프로젝트 환경 구축하기

이번 장은 도커(Docker)를 사용해 MySQL을 설치하고, 파이썬 노트북으로 MySQL과 연결하는 방법에 대해 학습한다. 더불어 이번 장에서 실습할 데이터를 다운받을 경로를 설명하고, 데이터에 대해 간단히 설명한다.

이 장에서 사용하는 데이터와 코드 원본은 아래 링크에서 확인할 수 있다.

- 데이터: [GitHub](https://github.com/hike-lab/address-data-guide/tree/main/chapter-5/data)
- 코드: [GitHub](https://github.com/hike-lab/address-data-guide/blob/main/chapter-5/5_%ED%8C%8C%EC%9D%B4%EC%8D%AC%EA%B3%BC_MySQL%EB%A1%9C_%EA%B5%AC%EC%B6%95%ED%95%98%EB%8A%94_%EC%A3%BC%EC%86%8C_%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B2%A0%EC%9D%B4%EC%8A%A4.ipynb)

## 도커로 MySQL 설치하기

도커는 어플리케이션을 패키징하고 배포하는 툴로, 컨테이너라는 가상 공간을 사용해서 어플리케이션과 실행헤 필요한 모든 것을 담아놓는다. 도커 형태로 어플리케이션을 패키징하게 되면, 개발 또는 운영 환경에 상관없이 안정적으로 어플리케이션을 운영할 수 있게 된다. 이번 장은 도커로 MySQL을 설치한다. 학습자의 개발 환경이 서로 상이한 상황에 있기 때문에 도커를 사용하는 것이 수월하다고 판단했다.

도커를 설치하는 방법은 윈도우와 맥, 리눅스 OS에 따라 다르다. 설치 방법은 다음의 공식문서에서 상세하게 설명하고 있다. 이 프로젝트는 `Docker version 20.10.12`로 진행한다.

- 윈도우 설치방법: [Install Docker Desktop on Windows](https://docs.docker.com/desktop/install/windows-install/)
- 맥 설치방법: [Install Docker Desktop on Mac](https://docs.docker.com/desktop/install/mac-install/)
- 리눅스 설치방법: [Install Docker Desktop on Linux](https://docs.docker.com/desktop/install/linux-install/)

### MySQL 이미지 가져오기

도커가 정상적으로 설치되었다면, 터미널을 이용해 MySQL 설치를 시작한다. 윈도우라면 명령 프롬프트나 PowerShell 등을 사용하면 되고, 맥이라면 터미널을 사용하면 된다. 터미널에서 다음의 코드를 실행해보자. 이 코드는 Docker Hub에 업로드되어 있는 최신 버전의 MySQL 이미지(image)를 가져온다. 도커에서 이미지는 컨테이너를 생성하는 데 사용되는 템플릿이다.

```bash
$ docker pull mysql
Using default tag: latest
latest: Pulling from library/mysql
ea4e27ae0b4c: Pull complete
837904302482: Pull complete
3c574b61b241: Pull complete
654fc4f3eb2d: Pull complete
32da9c2187e3: Pull complete
dc99c3c88bd6: Pull complete
970181cc0aa6: Pull complete
d77b716c39d5: Pull complete
9e650d7f9f83: Pull complete
acc21ff36b4b: Pull complete
Digest: sha256:ff5ab9cdce0b4c59704b4e2a09deed5ab8467be795e0ea20228b8528f53fcf82
Status: Downloaded newer image for mysql:latest
docker.io/library/mysql:latest
```

정상적으로 이미지를 가져왔다면, 위의 주석처리된 코드와 같이 나타난다. 다음의 코드는 다운로드 받은 도커 이미지의 목록을 보여준다. `mysql`이란 이름으로 가장 최신의 이미지를 가져온다.

```bash
$ docker images
REPOSITORY                      TAG               IMAGE ID       CREATED         SIZE
mysql                           latest            e68e2614955c   4 weeks ago     638MB
```

### Docker MySQL 실행하기

다음은 다운로드 받은 도커 이미지를 실행하는 방법이다. `docker run`은 이미지를 실행하는 기본적인 명령어이고, 안정적으로 도커 컨테이너를 실행하기 위해 여러가지 조건을 부여한다.

- `--name mysql`은 컨테이너 이름을 지정한다. 이 컨테이너의 이름은 `mysql`이 된다.
- `-p 3306:3306`은 호스트 시스템 포트인 3306를 컨테이너 내부 포트인 3306에 연결한다. 즉, 나의 컴퓨터에서 `localhost:3306`에 접속하면 컨테이너 내부의 MySQL 서버에 연결된다. 만약 로컬에서 이미 3306 포트가 사용된다면 `-p 3307:3306`과 같이 포트 번호를 변경할 수 있다.
- `-e MYSQL_ROOT_PASSWORD=root`는 MySQL root 사용자의 비밀번호를 root로 지정한다. 비밀번호는 반드시 다른 것으로 변경하여 사용하는 것이 좋다.
- `-v mysql-volume:/var/lib/mysql`는 `mysql-volume`이라는 이름의 볼륨을 컨테이너 내부 `/var/lib/mysql` 디렉토리와 연결한다. 개별적으로 볼륨을 생성하게 되면, 해당 컨테이너가 삭제되어도 볼륨은 삭제되지 않으니 데이터를 보존할 수 있다.
- `-d`는 컨테이너를 백그라운드에서 실행한다.
- `mysql:latest`는 `docker run`할 이미지의 이름과 태그를 작성한다.
- `--loose-local-infile=1`은 MySQL에서 `local infile` 기능을 사용하도록 설정한다. `local infile`은 MySQL에 파일을 삽입할 수 있는 명령어로, 서버 단에서 `local infile` 기능을 설정해주는 방법이다.

```bash
$ docker run --name mysql \\
    -p 3306:3306 \\
    -e MYSQL_ROOT_PASSWORD=root \\
    -v mysql-volume:/var/lib/mysql \\
    -d mysql:latest \\
    --loose-local-infile=1
```

정상적으로 도커 컨테이너가 실행됐다면 다음의 코드로 현재 돌아가는 컨테이너를 확인할 수 있다. 내가 지정한 옵션으로 컨테이너가 잘 돌아가고 있는지 확인해보자.

```bash
$ docker ps
CONTAINER ID   IMAGE          COMMAND                  CREATED          STATUS          PORTS                               NAMES
5dedb08a6941   mysql:latest   "docker-entrypoint.s…"   22 seconds ago   Up 22 seconds   0.0.0.0:3306->3306/tcp, 33060/tcp   mysql
```

혹시 컨테이너가 잘 만들어지지 않았다면, 다음의 과정으로 컨테이너를 중지하고 삭제할 수 있다. 우선 mysql 컨테이너를 중단하고, 이후에 mysql 컨테이너를 삭제한다.

```bash
$ docker stop mysql
$ docker rm mysql
```

### 컨테이너 내부의 MySQL 접속

컨테이너가 잘 돌아가고 있다면 컨테이너 내부에 있는 MySQL에 접속한다. 터미널에서 다음과 같이 입력하고, `mysql -u root -p`를 작성한다. `-u root`는 root 유저로 접속하는 것이므로 앞서 root 계정의 비밀번호로 설정한 값을 입력한다. 정상적으로 수행했다면 `mysql>`이 나타나면서 컨테이너 내부에 있는 MySQL에 접속한 것이다.

```bash
$ docker exec -it mysql bash
bash-4.4# mysql -u root -p
Enter password:
Welcome to the MySQL monitor.  Commands end with ; or \g.
Your MySQL connection id is 8
Server version: 8.3.0 MySQL Community Server - GPL

Copyright (c) 2000, 2024, Oracle and/or its affiliates.

Oracle is a registered trademark of Oracle Corporation and/or its
affiliates. Other names may be trademarks of their respective
owners.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.\
mysql>
```

MySQL은 SQL이란 질의 언어를 사용한다. `SHOW DATABASES;`는 현재 MySQL에 설치된 전체 데이터베이스를 보여준다. 아래 4가지가 기본적으로 설치되어 있는 데이터베이스다. SQL을 작성할 때 항상 마지막에 세미콜론(;)을 붙여야 한다.

```sql
mysql> SHOW DATABASES;
+--------------------+
| Database           |
+--------------------+
| information_schema |
| mysql              |
| performance_schema |
| sys                |
+--------------------+
4 rows in set (0.02 sec)
```

## 파이썬 노트북으로 MySQL 연결하기

MySQL이 잘 돌아가고 있다면 파이썬으로 MySQL에 접속할 수 있다. 이를 위해서 `pymysql`이란 모듈이 필요하다. 해당 모듈을 설치하기 전에 본인이 편한 파이썬 노트북 환경을 선택한다. [5.2장에서 설명한 VSCode](/contents/chapter-5/chapter-5-2.md)를 사용해도 되고, 쥬피터 노트북을 사용해도 된다. 파이썬 노트북 환경에서 다음과 같이 `pymysql`을 설치한다.

```py
!pip3 install pymysql
```

`pymysql`로 도커 컨테이너의 MySQL과 연결하는 방법은 다음과 같다. `init_connection()` 함수는 MySQL을 처음 연결할 때 사용하는 함수다. root 유저의 비밀번호를 변경했다면 `password="root"`에서 비밀번호를 수정한다. `sql` 변수는 앞서 컨테이너 내부에서 수행한 코드와 동일하다. `cur.execute(sql)`에서 SQL 쿼리를 실행하고, 결과를 출력한다.

```py
import pymysql

def init_connection():
    connection = pymysql.connect(
        host="localhost",
        port=3306,
        user="root",
        password="root",
        cursorclass=pymysql.cursors.DictCursor,
        charset="utf8",
    )
    return connection

sql = "SHOW DATABASES;"
conn = init_connection()

with conn:
    with conn.cursor() as cur:
        cur.execute(sql)
        for data in cur:
            print(data)
# {'Database': 'information_schema'}
# {'Database': 'mysql'}
# {'Database': 'performance_schema'}
# {'Database': 'sys'}
```

위의 코드에서 주석처리 된 것처럼 나온다면 정상적으로 잘 연결된 것이다. 파이썬에서 MySQL과 연결하게 되면, 항상 접속한 상태이므로 `conn.close()`을 통해 연결을 닫아줘야 한다.

```py
conn.close()
```

## 데이터 다운로드

이번 장에서 사용할 실습 데이터는 주소기반산업지원서비스의 도로명주소 한글을 사용한다. 해당 데이터는 [3.3장](/contents/chapter-3/chapter-3-3.md)에서 자세하게 설명하고 있으니 참고하면 된다. 데이터는 2024년 1월 전체자료분을 사용하고, 도로명주소 테이블 `rnaddrkor`과 관련지번 테이블 `jibun_rnaddrkor`을 모두 사용한다. 실습에 사용한 데이터는 [여기](https://github.com/hike-lab/address-data-guide/tree/main/chapter-5/data)에서 `rnaddrkor.zip` 파일을 다운로드 받을 수 있다.
