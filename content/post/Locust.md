---
title: "Locust : 파이썬 기반 오픈 소스 로드 테스트"
date: 2021-04-25T14:56:52+09:00
draft: false
categories: ["open-source"]
tags: ["python", "loadtest", "locust", "opensource"]
---

# Locust
Locust는 사용이 쉽고, 스크립트 가능하고, 확장 가능한 성능 테스트 도구이다.    
유저들의 행동을 파이썬 코드를 사용해서 정의할 수 있다. 

[Locust 공식 GitHub ](https://github.com/locustio/locust)
## Features

### 평범한 Python으로 사용자 테스트 시나리오 작성 
Locust는 경량 코루틴인 greenlet 내에서 모든 유저를 실행한다. 그렇기 때문에 콜백이나 다른 메커니즘을 사용하지 않고 일반 python 코드와 같은 테스트를 작성할 수 있다. 

### 분산 & 확장 가능 - 수 십만의 유저 지원
Locust를 사용하면 분산된 부하 테스트를 여러 시스템에 쉽게 동작 시킬 수 있다. gevent를 사용하는 이벤트 기반이기 때문에, 단일 프로세스에서도 수 천의 동접 유저 처리를 할 수 있다. 더 많은 요청을 수행할 수 있는 다른 도구가 있을 수 있지만, Locust의 각 유저의 낮은 오버헤드는 동시 부하를 테스트하는데 적합하다.

### Web-based UI
Locust는 플라스크를 사용해서 webUI를 서빙한다. 따라서 web endpoint를 추가하기 쉽다. Flask Blueprints와 templates를 사용할 수 있다. 
Locust는 유저 친화적인 웹 인터페이스로 테스트를 실시간으로 보여 준다. UI 없이도 사용해서 CI/CD 테스트에서도 쉽게 사용할 수 있다.
```python
from locust import events

@events.init.add_listener
def on_locust_init(web_ui, **kw):
    @web_ui.app.route('/added_page")
    def my_added_page():
        return "Another page"
```

### Can test any system


## How to ~
### 설치 하기
| pip 을 사용한 설치
```sh
$ pip3 install locust
$ locust -v 
```
### 실행 하기
| 특정 Path에 위치한 Locust 파일을 실행하기
- --master
- --worker
- --headless
```sh
$ locust -f locust_files/my_locust_file.py # http://127.0.0.1:8089
```
| Docker-Compose를 사용해 실행하기
```sh
$ docker-compose up --scale worker=4
```
## Writing a locusfile
locustfile은 일반적인 python 파일이다. 최소 User 클래스를 상속 받는 클래스 하나면 필요하다.

### User class
Locust는 각 시뮬레이트 될 각 유저 마다 User 클래스의 인스턴스를 생성한다.    
User 클래스에서 정의해줘야 하는 공통 attribute들이 존재한다.   

 <br>wait_time attribute</br>

 ```Users``` 클래스의 ```wait_time``` 메소드는 optional attribute로 가상 유저들이 대기해야하는 태스크들 사이의 시간이 필요할 때 사용한다. ```wait_time```이 정의되지 않으면 새로운 태스크는 대기 없이 바로 실행된다.
 ```python
 from locust import User, task, between

 class MyUser(User):
    @task
    def my_task(self):
         print("Executing my_task")
    
    wait_time = between(0.5, 10)
 ```

<br>weight attritubute</br>

하나 이상의 User 클래스가 파일에 존재하거나, User 클래스들이 커맨드 라인에 구체화 되지 않았을 때 Locust는 각 User 클래스들을 동일한 수 만큼 생성한다. 
동일한 파일 내에서 어떤 User 클래스를 특정할 지, 커맨드 라인에서 정할 수 있다.
```sh
$ locust -f locust_file.py WebUser Mobile User
```
weight attribute를 클래스에서 사용해서 더 많은 유저들로 시뮬레이트 할 수 있다.
```python
class WebUser(User):
    weight = 3
    ...

class MobileUser(User):
    weight = 1
    ...
```

<br>host attribute</br>

커맨드 라인에서 ```--host``` 옵션을 사용할 수 있으나 WebUI에서 입력할 수 있다.

<br>tasks attribute</br>

User 클래스는 메소드로 정의된 태스크를 ```@task``` 데코레이터를 사용해서 선언할 수 있다.    
tasks attribute를 사용해서도 할 수 있다.

<br>environment attribute</br>
    ...

<br>on_start & on_stop 메소드</br>
User ( 그리고 TaskSets ) 가 선언할 수 있다. 
- User : ```on_start``` 메소드는 running 시작 시에, ```on_stop``` 메소드는 running 이 멈춘 뒤호출 된다.
- TaskSet : ```on_start``` 메소드는 TaskSet을 가상 유저가 실행 시에, ```on_stop```은 가상 유저의 TaskSet 실행이 멈추거나, 유저가 죽을 때 호출 된다.

### Tasks
<br> tasks attribute </br>

task를 선언하는 다른 방법 중 하나이다.
```python
from locust import User, constant

def my_task(user)
    pass

class MyUser(User):
    tasks = [my_task]
    wait_time = constant(1)
```
<br> @task decorator</br>
<br> @tag decorator</br>

### Events
테스트의 한 파트로 일부 설정 코드를 사용하기에 locustfile에 모듈 레벨로 두는 것으로 충분하지만,    
실행 중에 특정 시간에 작업을 하는 경우에는 Locust에서 제공하는 event 훅을 사용하면 된다.

<br>test_start & test_stop</br>
부하 테스트 start 혹은 stop 시, 일부 코드를 실행해야 하는 경우, ```test_start```와 ```test_stop``` 이벤트를 사용해야 한다. locust 모듈 수준에서 이러한 이벤트에 대한 리스너를 설정할 수 있다.
```python
from locust import events

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("A new test is starting")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("A new test is ending")
```

<br>init</br>
```init``` 이벤트는 각 Locust 프로세스가 시작 될 때, 트리거 된다.    
분산 모드에서 각 워커 프로세스가 초기화 시에 뭔가 해야 할 일이 필요한 경우에 특히 유용하다. 
```python
from locust import events
from locust.runner import MasterRunner

@events.init.add_listener
def on_locust_init(environment, **kwargs):
    if isinstance(environment.runner, MasterRunner):
        print("I'm on master node")
    else:
        print("I'm on worker or standalone node")
```

### HttpUser class
<br>Validating responses</br>
리쿼스트 들은 HTTP response code 가 OK 시에 성공으로 판단된다. (< 400)     
그러나 추가 validation이 필요한 경우에 사용 될 수 있다.
- catch_response 파라미터
- with 구문
- response.failure() 호출
```python
with self.client.get("/", catch_response=True) as response:
    if response.text != "Success":
        response.failure("Got wrong response")
    elif response.elapsed.total_seconds() > 0.5:
        response.failure("Request took too long")
```

```python
with self.client.get("/does_not_exist/", catch_response=True) as response
    if response.status_code == 404:
        response.success()
```
```python
from json import JSONDecodeError
...
with self.client.post("/", json={"foo":42, "bar":None}, catch_response=True) as response:
    try:
        if response.json()['greeting'] != 'hello':
            response.failure('Did not get expected value in greeting')
    except JSONDecodeError:
        response.failure("Response could not be decoded as JSON")
    except KeyError:
        response.failure("Response did not contain expected key 'greeting'")
```
```python
for i in range(10):
    self.client.get("/blog?id=%i" % i, name="/blog?id=[id]")
```





## Advanced
### Locust의 성능을 향상 시키기 faster HTTP client 
Locust의 기본 HTTP client는 python-request이다. 잘 관리되는 파이썬 패키지이고 사용하는 것이 권장 된다.     
만약 굉장히 거대한 규모의 테스트를 계획한다면, 다른 HTTP Client를 대안으로 사용할 수 있다. ```FastHttpUser```를 사용해서 geventhttpclient로 requests를 하는 것이다.
```python
    class FastLargeScaleUser(FastHttpUser):
        wait_time = between(2, 5)

        @task 
        def index(self):
            response = self.client.get("/")
```
### 커스텀 load 형태 생성 
load spike를 생성하거나 지정한 시간에 늘리거나 줄이는 것을 원한다면, ```LoadTestShape``` 클래스를 사용해서 컨트롤 할 수 있다.    
해당 클래스에서 ```tick()``` 메소드를 정의할 수 있는데 Locust는 메소드를 초당 한번 호출한다. 
```python
    class MyCustomShape(LoadTestShape):
        time_limit = 600
        spawn_rate = 20 

        def tick(self):
            run_time = self.get_run_time() # check time how long the test run for
            # user_count = get_current_user_count()

            if run_time < self.time_limit:
                user_count = round(run_time, -2)
                return (user_count, spawn_rate)
            return None
```
### 로깅
Locust는 python의 빌트인 로깅 프레임워크를 사용해서 로그를 핸들링한다.    

### User class
> class User(environment)

생성될 'user'를 나타내며 로드 테스트할 시스템을 공격한다. 이 유저의 행동은 task로 정의된다.   
Tasks는 ```@task decorator``` 메소드를 사용하거나 ```tasks attribute```를 세팅함으로 선언된다.

- abstract() : True면, 클래스가 서브 클래스가 된다. 테스트 중에 유저들을 생성하지 않는다.
- on_start() : User가 running을 시작할 때 호출된다.
- on_stop() : User가 running을 멈출 때 호출된다.
- tasks() : python으로 호출 가능한 태스크셋 클래스들의 Collection
    - list : 태스크가 랜덤으로 선택되어 동작된다.
    - 두개 이상의 tuple(callable, int)로 구성된 list나 dict 시에 int 값으로 가중되게 선택되어 동작한다.

- wait() : User.wait_time 함수로 정의된 시간동안 동작 중인 유저를 sleep
- wait_time() : locusts 태스크들 실행 간 시간을 반환하는 메소드

### TaskSet class
> class TaskSet(parent)

User가 실행할 태스크들의 셋을 정의하는 클래스
태스크셋이 동작하기 시작했을 때, tasks attribute로 부터 task를 선택하고, 실행하고, wait_time 함수로 반환된 초 만큼 sleep 한다.     
태스크셋은 중첩될 수 있다. 

### task decorator
> task(weight=1)

User 또는 TaskSet에 대한 작업을 클래스에서 in-line으로 선언 할 수있는 편리한 데코레이터로 사용된다.
### tag decorator
> tag(*tag)

주어진 태그명으로 tasks와 TaskSets를 태깅하는 데코레이터
```python
class ForumPage(TaskSet):
    @tag('thread')
    @task(100)
    def read_thread(self):
        pass 
    
    @tag('thread')
    @tag('post')
    @task(7)
    def create_thread(self):
        pass
```

### SequentialTaskSet class
> SequentialTaskSet(*args, **kwargs)

User가 실행할 태스크 들의 시퀀스를 선언하는 클래스    
TaskSet 클래스 처럼 동작하지만, task weight가 무시된다. 모든 태스크들이 순서대로 실행된다.


### 빌트인 wait_time 함수들
> between(min_wait, max_wait)

min_wait ~ max_wait 간 랜덤 넘버를 반환한다.

> constant(wait_time)

wait_time으로 고정된 넘버를 반환한다.

> constant_pacing(wait_time)

Task의 실행 시간을 트래킹하는 함수를 반환하고 호출 마다 Task 실행 시간 사이의 총 시간을 wait_time 인수에 지정된 시간과 동일하게 만들려고 하는 대기 시간을 반환한다.
만약 task 실행 시간이 선언된 wait_time 초과 되면, 다음 task 실행 시 까지 wait가 0가 된다.
```python
    class MyUser(User):
        wait_time = between(3.0, 10.5)
        #wait_time = constant(3)
        #wait_time = constant_pacing(1)

        @task
        def my_task(self):
            time.sleep(random.random())
```

### HttpSession class 
WHAT?

### Response class
이 클래스는 python-requests 라이브러리 내에 위치, request 문서에서 확인할 수 있음.
HTTP request에 대한 서버의 응답을 담고 있는 ```Response``` object

> class Response
- property
    - apparent_encoding   
    - is_permanent_redirect    
    - is_redirect   
    - content ( 바이트 형태의 response content )    
- close()   
- cookies = None    
- elapsed = None ( request를 보내고 response 도달 까지의 시간 )    
- encoding = None ( r.text를 디코딩 하기 위한 인코딩 )    
- headers = None    
- history = None    
- iter_content(chunk_size=1, decode_unicode=False)    
    -    response 데이터에 대해 iterate stream=True일 때, 긴 response에 대해 content를 메모리를  한번에 올리지 않는다. chunk_size 만큼 메모리에 올리고, 지정된 unicode로 decode )        
- iter_lines(chunk_size=1, decode_unicode=False, delimeter=None)
    - 한번에 한 라인
- json(**kwargs)
    - json 인코딩된 response의 content를 반환한다.
        - Parameters
        - Raises
    - property 
        - links
        - next
        - ok 
        - text
    - raise_for_status()
    - raw = None
    - reason = None
    - request = None 
    - status_code = None 
    - url = None

### ResponseContextManager class
> class ResponseContextManager(response, request_access, request_failure)

HTTP 요청이 Locust 통계에서 성공 또는 실패로 표시되어야 하는지 여부를 수동으로 제어하는 기능을 제공하는 context manager 역할도 하는 response class

이 클래스는 ```Response```의 서브 클래스로 success와 failure 두가지 메소드가 추가 되었다.

> failure(exc)
```python
    with self.client.get("/", catch_response=True) as response:
        if response.content == b"":
            response.failure("No data")
```
> success(exc)
```python
    with self.client.get("/does/not/exist", catch_response=True) as response:
        if response.status_code == 404:
            response.success()
```
    
