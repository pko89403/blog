---
title: "FastAPI의 async def와 def에 대해서 공부해보자"
date: 2021-09-05T18:19:34+09:00
draft: false
categories: ["python"]
tags: ["fastapi", "async", "python", "api"]
---

FastAPI의 async def에 관해 작성된 페이지인 [Concurreny and async/await](https://fastapi.tiangolo.com/async/) 를 보았다.

우선 기본적으로 FastAPI는 `async def`를 꼭 쓸 필요는 없다. def 만으로도 비동기 처리 되도록 FastAPI 프레임 워크로 구현하고 있다.

![Untitled](FastAPI%E1%84%8B%E1%85%B4%20async%20def%E1%84%8B%E1%85%AA%20def%200f2777788c114692bdd208c5012a584b/Untitled.png)
{{< figure src="/images/fastapiasync/Untitled.png" title="" >}}


다음과 같이 await 로 호출하도록 가이드를 하는 서드 파티 라이브러리를 사용하는 경우

```python
results = await some_library()
```

다음과 같이 선언해서 사용하라고 한다.

```python
@app.get('/')
async def read_results():
	results = await some_library()
	return results
```

await 사용을 지원하지 않는 서드 파티 라이브러리를 사용하는 경우, 그냥 def를 사용해서 사용하라고 한다.

```python
@app.get('/')
def results():
	results = some_library()
	return results
```

어플리케이션이 다른것과 통신을 하지 않거나 응답을 기다리지 않는다면,  `async def`를 사용하라고 한다. 

### Technical Details

파이썬은 "비동기 코드"를 코루틴(coroutines)를 `async` 와 `await` 으로 지원하고 있다.

비동기식 코드는 언어가 프로그램에게 코드의 특정 지점에서 다른 작업이 다른 곳에서 끝날 때까지 기다려야 함을 선언하는 방법을 가지고 있음을 의미한다.

코루틴은  `async def` 함수로 반환 된 것을 용어로 표현한 것으로, 파이썬은 함수 같이 코루틴이 시작될 수 있고 어느 시점에 끝날 지 알고, await가 있을 때마다 내부적으로 일시 중지 될 수 있다는 것도 알고 있다.

다른 작업을 기다린다는 의미를 다음과 같은 표현으로 나열할 수 있다.

- 클라이언트가 네트워크를 통해 보낼 데이터
- 시스템이 읽고 프로그램에 제공할 디스크의 내용
- 프로그램이 디스크에 쓰기 위해 시스템에 제공한 내용
- 원격 API 작업
- 완료할 DB 작업
- 결과를 반환하는 DB 쿼리

기존의 Flask와 Django는 Python의 새로운 비동기 기능이 나오기 이전에 나왔다. 따라서 이전 방식의 비동기 실행을 지원한다.

## FastAPI << Starlette << Uvicorn << Cython

ASGI(Asynchronous Web Python)의 주요 스펙은 Djang에서 개발되었지만, WebSocket에 대한 지원이 추가되었다. FastAPI는 병렬성과 비동기성 모두를 지원하기 때문에 성능 향상을 가질 수 있었다고 한다.

FastAPI는 Starlette를 한번 감싸서 개발한 ASGI 프레임워크이다. 그렇기 때문에 Starlette를 직접 사용하는 것보다는 성능이 떨어지지만, 개발 속도를 올릴 수 있다. 

Starlette는 내부적으로 uvicorn을 사용하는데 uvicorn은 uvloops와 httptools를 사용하는 초고속 ASGI 서버이다. 

그리고 uvloop의 성능 비밀은 libuv와 Cython에 있다고 한다. uvicorn 실행 시에 `—loop` 옵션으로 `asyncio` 와 `uvloop`를 선택할 수 있다. auto 도 있는데 패키지가 설치 되어있다는 기준으로 선택한다고 한다.

## Very Technical Details

`async def` 대신 `def`로 경로 함수를 선언하면, 외부 스레드 풀에서 실행 되어서 다이렉트로 호출 되지 않아. 서버가 블록 되지 않는다.

I/O 바운드가 발생하지 않는 경우 또는 async/await를 지원하는 라이브러리를 사용하는 경우, `async def`를 사용하는 것이 좋다.

단, async/await 를 지원하는 라이브러리가 없는 경우에는 `async def`를 사용할 필요가 없다.

디펜던시에서도 동일하게 적용된다. 만약 디펜던시가 `async def` 대신 `def` 면, 외부 스레드 풀에서 실행된다.

`def / async def` 를 혼합 해서 사용해도 def 부분은 외부 쓰레드 풀에서 실행된다.

## 참고 사이트

- [https://fastapi.tiangolo.com/async/](https://fastapi.tiangolo.com/async/)
- [https://qiita.com/ffggss/items/e4c06f86fb28a62948e0](https://qiita.com/ffggss/items/e4c06f86fb28a62948e0)
- [https://jybaek.tistory.com/890](https://jybaek.tistory.com/890)
- [https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=parkjy76&logNo=221983329279](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=parkjy76&logNo=221983329279)