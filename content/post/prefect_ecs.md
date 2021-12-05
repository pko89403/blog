---
title: "워크플로우 엔진 PREFECT : ECS Agent 로 Prefect 환경 구축하기"
date: 2021-12-05T22:33:51+09:00
draft: false
categories: ["Workflow"]
tags: ["AWS", "ECS", "Prefect", "Workflow", "Data Pipelining"]
---

Prefect를 AWS 클라우드 환경에서 구성했다. 다양한 방법으로 Prefect를 구성할 수 있지만,   
EC2 Instance 상에 Prefect Backend의 오픈소스인 Prefect Server를 실행시키고 ECS Cluster를 생성한 다음 ECS Agent를 사용했다.    
Prefect Cloud는 유료이기 때문에 사용하지 않았고 오픈소스인 Prefect Server로 Prefect Core를 구성하고,    
K8S Agent로 구성해도 되지만, 거대한 워크플로우 관리를 할 것이 아니기 때문에 ECS Agent로 구성했다.

따라서 이 글에서 다룰 내용은 다음과 같다.
- EC2에서 Prefect Server 실행 하기
- EC2에서 ECS Agent 실행 하기
- ECS Service로 ECS Agent를 실행 하기 ( Production )

Prefect Cloud를 사용하지 않은 오픈 소스인 Prefect Server를 사용해서 구성했다.

# 1. Prefect Architecture

## 1.1. Prefect Server

Prefect Server는 데이터를 직접 유지하지 않고 메타 데이터로만 동작한다. Prefect를 구성하는 각 마이크로서비스는 다음과 같다.

{{< figure src="/images/Prefect_ECS/0.png" title="Prefect Architecture" >}}

- UI - 메타 데이터를 변경하고 쿼리를 하기 위한 대시보드 제공
- Apollo - 서버와 연동된 메인 엔드포인트. Agent와 UI가 접근 해야한다
- GraphQL - GraphQL 뮤테이션을 노출하는 서버 비즈니스 로직
- Hasura - DB에서 메타데이터를 쿼리하는 GraphQL API
- PostgreSQL - 메타데이터가 저장되는 DB
- Towl - 서버 유지 관리를 위한 유틸리티 실행
    - Scheduler - Flow를 예약하고 실행
    - Zombie Killer - Tasks의 실패를 표현
    - Lazarus - Flow 실행을 재 스케줄링

## 1.2. Prefect Agent

Prefect의 Agent는 Prefect의 워크플로우인 Flow를 선택하고 실행한다.

# 2. Prefect Server

Prefect Server는 오픈 소스로 Prefect의 Flow를 실행하고 모니터링을 쉽게 만들어 준다. 여러가지 서비스의 조합으로 Flow 실행과 현재 상태에 대한 영구적인 기록, 비동기 스케줄링과 알림을 제공한다. 특히, 아래의 서비스를 기본적으로 제공한다.

- 영구 메타 데이터 DB
- 높은 수준의 확장성을 제공하는 스케줄러
- Query 작성 및 작업 트리거를 위한 표현형 GraphQL API ( 이벤트 기반 Flow 실행 )
- 다양한 실행 환경에서 Flow를 스케줄링하고 오케스트레이션 할 수 있는 Host 프로세스와 Execute 프로세스를 분리하는 유니크한 설계
- Flow가 성공하거나 최소한 성공적인 실패를 보장하는 서비스
- 모든 기능을 갖춘 UI

Prefect Server 개발은 전적으로 오픈 소스이고, 코드는 두 리포지토리에 있다. 

- [Prefect Server Github Repository](https://github.com/PrefectHQ/Server) : Prefect Server에서 실행되는 모든 서비스 코드
- [Prefect UI Github Repository](https://github.com/PrefectHQ/ui) : Prefect Server와 Prefect Cloud에서 실행되는 코드

## 2.1. Prefect Server 실행 ( EC2 )

AWS EC2 환경에서 Prefect Server를 구축하기로 했다. EC2를 생성하고 ssh를 통해 EC2에 접속한다. 그리고 아래 커맨드 들을 입력해서 기본적으로 필요한 Python과 Docker와 Docker-Compose를 설치 했다. 

추가로 Security Group에서 UI ( :8080 )와 Backend ( :4200 )을 오픈해야한다.

```bash
sudo yum install -y python3 pip3 # 파이썬 설치
sudo amazon-linux-extras install docker # 도커 설치
sudo service docker start # 도커 시작
sudo usermod -a -G docker ec2-user # 도커 권한 추가
sudo curl -L https://github.com/docker/compose/releases/download/v2.1.1/docker-compose-linux-x86_64 -o /usr/local/bin/docker-compose # 도커 컴포즈
sudo chmod +x /usr/local/bin/docker-compose # 도커 컴포즈 권한 추가
```

Prefect를 설치하기 위해서는 파이썬 3.6 이상의 버전이 필요하다. 3.9 버전에 대한 완벽한 지원은 아직 안된다고 한다. Anaconda로 설치하는 것을 공식 문서에서는 권장하고 있다. 여기에서 사용하고 있는 Prefect의 버전은 0.15.9 다.

```bash
conda install -c conda-forge prefect
```

엑스트라 패키지가 포함된 버전을 설치할 수도 있다. 지원하는 엑스트라 패키지는 다음과 같다.

- all_extras : 모든 사용 가능한 디펜던시 도구
- dev : Prefect 자체를 개발하기 위한 도구
- templates : 문자열 템플릿 작업을 위한 도구
- viz : Perfect Flow 시각화를 위한 도구
- aws : AWS와 상호 작용하기 위한 도구
- azure : Azure과 상호 작용하기 위한 도구
- google : GCP와 상호 작용하기 위한 도구
- kubernetes : K8S API 객체와 상호 작용하기 위한 도구
- twitter : Twitter API와 상호 작용하기 위한 도구
- airtable : Airtable API와 상호 작용하기 위한 도구
- spacy : Spacy와 상호 작용하여 NLP 파이프라인을 구축하기 위한 도구
- redis : Redis DB와 상호작용 하기 위한 도구

EC2에서 따로 가상환경을 설정하지 않고 Prefect Server 만 실행할 것이라 pip를 사용해서 설치를 진행한다.

```bash
pip install "prefect[aws]"
```

아래의 커맨드를 입력해서 Prefect가 Prefect Server를 사용할 수 있도록 구성한다. 

```bash
prefect backend server
```

그리고 root/.prefect 경로에 config.toml 파일을 생성한 후 Prefect Core가 사용할 호스트명을 입력한다.

```bash
[server]
[server.ui]
apollo_url = "{Prefect Server Host}:4200/graphql"

[cloud]
api = "{Prefect Server Host}:4200"
```

Prefect Server를 아래의 커맨드로 실행한다. 그러면 Docker-Compose가 실행되며 Prefect Server가 구성된다. 

```bash
prefect server start --expose -d # AWS GCP에서 사용할 때
```

EC2의 `IP:8080`으로 접속하게 되면 Prefect UI에 접속할 수 있다. 아래와 같은 화면이 보이게 된다.

{{< figure src="/images/Prefect_ECS/1.png" title="Prefect UI 화면" >}}

# 3. Prefect Agents

Prefect Agent는 Flow를 오케스트레이팅하는 경량 프로세스다. Prefect Server의 Apollo에 접근해서 연동 된 다음에 사용이 가능하다. 

에이전트는 Flow를 실행하고 모니터링 하는 역할을 한다. 스케줄링된 Flow를 동작 시키기 위해 Prefect API에 Query하고 Query를 위한 리소스를 할당한다. Agent는 Prefect API에 요청을 보내기만 하고 요청을 받지는 않는다.  Prefect에서 지원하는 Agent의 유형은 아래와 같다.

- Local Agent ( UniversalRun + LocalRun )
- Docker Agent
- Kubernetes Agent
- GCP Vertex
- AWS ECS ( Fargate Deprecated )

사용할 수 있는 Agent는 UI를 통해 확인 할 수 있다.

{{< figure src="/images/Prefect_ECS/2.png" title="Prefect UI:Agent 화면" >}}

## 3.1. Local Agent

Prefect Agent를 로컬 프로세스로 구성해서 Flow를 실행한다. 가벼운 워크플로우나 로컬에서 Flow를 테스트하는데 사용하는 것이 적절하다. 구성하는 방법은 아래의 커맨드를 실행하면 된다.

```bash
prefect agent local start
prefect agent local install # supervisor 를 사용하는 것을 지원한다
prefect agent local start --api {Prefect Server Host}:4200 --label {Agent가 사용할 Label}
```

## 3.2. Docker Agent

개별 Docker Container에서 Flow를 실행한다. Local Agent에 비해 더 많은 격리성과 제어를 제공한다.

로컬에서 구성하는 방법은 아래와 같다.

```bash
prefect agent docker start --api {Prefect Server Host}:4200 --label {Agent가 사용할 Label}
```

EC2에서는 ECR을 사용하기 위해서 아래와 같이 구성해서 실행할 수 있다. 

```bash
prefect agent docker start --api {Prefect Server Host}:4200 \
--label {Agent가 사용할 Label} \
--env AWS_ACCESS_KEY_ID={} \
--env AWS_SECRET_ACCESS_KEY={} \
--env AWS_REGION=ap-northeast-2
```

아래와 같이 개발한 Flow의 이미지를 빌드하고 Flow를 Server에 등록하고 ECR에 푸쉬해 Flow를 실행할 수 있다. ( Flow 개발과 Flow를 등록하는 방법은 다음 글에서 다룰 예정이어서 여기서는 다루지 않겠다 )

```bash
docker build -t {Flow 이미지 명} .
docker run -it --network host {Flow 이미지 명} python -m {Flow 명}
docker tag {Flow 이미지 명}:{Flow 이미지 태그} {ECR Repository 명}/{Flow 이미지 명}:{Flow 이미지 태그}
docker push {ECR Repository 명}/{Flow 이미지 명}:{Flow 이미지 태그}
```

## 3.3. ECS Agent

ECS는 AWS에서 제공하고 있는 클러스터 관리 서비스중 하나다. Prefect ECS Agent는 ECS Tasks로 Flow를 실행하는 방법이다.

ECS Agent 사용 시, 고려해야할 점이 몇가지가 있다.

- —launch-type : ECS에 Flow를 배포해서 실행하는 타입 ( EC2, Fargate )
- —task-role-arn : ECS Task가 실행하면서 코드에서 AWS API를 호출하기 위한 IAM Role
- —execution-role-arn : Task를 실행하기 위해 초기 AWS API를 호출하기 위한 IAM Role

추가적으로 ECS 네트워크 모드( host, bridge, awsvpc )에 대해 잘 알 수 있다면 더 자세한 설정이 가능해진다.

### 3.3.1. EC2를 사용해서 ECS Agent를 사용

ECS Cluster를 별도의 설정 없이 생성하고 EC2를 사용해서 ECS Agent를 간단하게 사용할 수 있다. 물론 ~/.prefect/config.toml 파일을 이용해서 Prefect Server의 Host에 대해 명시해야한다.

```bash
[cloud]
api = "{Prefect Server Host}:4200"
```

기본적으로 사용할 launch-type이나 task-role-arn이나 execution-role을 아래의 커맨드로 정의해서 기본적인 IAM과 Task Definition을 사용 할 수 있지만, 개인적인 판단으로는 Flow를 register 하는 당시에 코드를 통해 선택할 수 있는데 그 방법이 더 좋다고 생각한다.

```bash
prefect agent ecs start --cluster {ECS 클러스터 ARN} \
--label {Agent가 사용할 Label}  > /dev/null 2>&1 &
```

```bash
prefect agent ecs start \
--cluster {ECS 클러스터 ARN} \
--execution-role-arn {Execution Role을 위해 생성한 IAM ARN} \
--task-role-arn {Task Role을 위해 생성한 IAM ARN} \
--label {Agent가 사용할 Label} \
--launch-type {ECS Task의 실행 타입(EC2|FARGATE)} > /dev/null 2>&1 &
```

### 3.3.2. ECS Service를 사용해서 ECS Agent를 사용

하지만 여기 까지는 Production으로 사용하기에는 많이 부족해 보인다. 그래서 좀 더 리서치를 했다. ECS Fargate를 Spot-Instance로 사용하고 Agent를 Fargate로 Task Runner를 Fargate_SPOT로 사용하는 방법으로 비용을 줄이고 좀 더 안정적으로 사용하는 방법을 찾아 실제로 테스트한 결과를 기록으로 남긴다.

아래와 같이 ECS 클러스터를 생성한다. 

- FARGATE : ECS Service로 ECS Agent가 실행될 것이다
- FARGATE_SPOT : ECS Task Runner가 실행 될 것이다

```bash
export AWS_REGION=ap-northeast-2
export ECS_CLUSTSTER_NAME=prefect-ecs-cluster 

aws ecs create-cluster --cluster-name $ECS_CLUSTER_NAME \
--capacity-providers FARGATE_SPOT FARGATE \
--default-capacity-provider-strategy \
capacityProvider=FARGATE_SPOT,weight=3 \
capacityProvider=FARGATE,base=1,weight=2 \
--region $AWS_REGION
```

ECS에서 Flow를 실행 시에 Log를 모니터링 해야 하기 때문에 별도의 LOG_GROUP을 생성한다.

```bash
export ECS_LOG_GROUP_NAME=/ecs/prefect-ecs-agent
aws logs create-log-group \
--log-group-name $ECS_LOG_GROUP_NAME \
--region $AWS_REGION
```

문제가 하나 있는데 ECS Agent를 등록하기 위해서는 공식적인 가이드는 Prefect Backend로 Server로 사용할 때의 가이드는 없고, 유료 서비스인 Cloud를 사용하는 방법에 대해서만 가이드 한다. 

EC2로 실행하고 있는 Prefect Server의 Host를 등록하기 위해 아래의 환경 변수들을 여러가지 방법을 테스트 했지만 성공하지는 못했다.

```json
{"name": "PREFECT__BACKEND", "value": backend}
{"name": "PREFECT__CLOUD__API", "value": prefect.config.cloud.api}
{"name": "PREFECT__CLOUD__AGENT__LABELS", "value": "[]"}
{"name": "PREFECT__CLOUD__USE_LOCAL_SECRETS", "value": "false"}
{"name": "PREFECT__CLOUD__SEND_FLOW_RUN_LOGS", "value": "true"}
{"name": "PREFECT__LOGGING__LEVEL", "value": "INFO"}
{"name": "PREFECT__ENGINE__FLOW_RUNNER__DEFAULT_CLASS", "value": "prefect.engine.cloud.CloudFlowRunner",}
{"name": "PREFECT__ENGINE__TASK_RUNNER__DEFAULT_CLASS", "value": "prefect.engine.cloud.CloudTaskRunner",}
{"name": "PREFECT__LOGGING__LOG_TO_CLOUD", "value": "true"}
```

따라서 Prefect의 Agent를 등록할 ECS Service의 Docker 이미지를 config.toml 파일을 추가하는 형태로 만들어서 아래와 같이 사용했다.

root/.prefect/config.toml 파일을 미리 작성한다.

```bash
[cloud]
api = "{Prefect Backend Server HostName}:4200"
```

Custom 하게 Prefect ECS Agent에서 사용할 Docker 이미지의 Dockerfile를 작성한 후, 빌드하고 Docker Registry에 푸시한다. 

```docker
FROM prefecthq/prefect:latest
RUN prefect backend server 

ADD agent_config.toml /root/.prefect/config.toml
```

ECS에서 사용할 Task Execution Role과 Task Role을 생성한다.    

Task Execution IAM Role
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ecr:GetAuthorizationToken",
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "*"
        }
    ]
}
```

Task IAM Role
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "iam:PassRole",
                "ec2:AuthorizeSecurityGroupIngress",
                "ec2:DescribeSecurityGroups",
                "ec2:DescribeVpcs",
                "ec2:DeleteSecurityGroup",
                "ec2:DescribeSubnets"
                "ec2:CreateSecurityGroup",
                "ec2:DescribeNetworkInterfaces",
                "ec2:CreateTags",
                "s3:*",
                "ecs:DescribeTaskDefinition",
                "ecs:DeregisterTaskDefinition",
                "ecs:DescribeClusters",
                "ecs:ListAccountSettings",
                "ecs:StopTask",
                "ecs:DescribeTasks",
                "ecs:ListTaskDefinitions",
                "ecs:ListClusters",
                "ecs:RunTask",
                "ecs:RegisterTaskDefinition",
                "ecs:CreateCluster",
                "ecs:DeleteCluster",
                "logs:CreateLogStream",
                "logs:GetLogEvents",
                "logs:DescribeLogGroups",
                "logs:PutLogEvents",
            ],
            "Resource": "*"
        }
    ]
}
```

ECS Service를 생성할 Task Definition를 등록하고 ECS 클러스터에 Service를 생성한다.
```json
{
  "executionRoleArn": "{생성 Task-Execution-IAM-Role-ARN}",
  "containerDefinitions": [
    {
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/prefect-ecs-agent",
          "awslogs-region": "ap-northeast-2",
          "awslogs-create-group": "true",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "command": [
        "prefect",
        "agent",
        "ecs",
        "start"
      ],
      "environment": [
        {
          "name": "PREFECT__CLOUD__AGENT__LABELS",
          "value": "[{Agent가 사용할 Label}]"
        },
        {
          "name": "PREFECT__CLOUD__AGENT__LEVEL",
          "value": "INFO"
        }
      ],
      "image": "{Custom 하게 생성한 Prefect ECS Agent용 도커 이미지}",
      "name": "prefect-ecs-agent-container"
    }
  ],
  "memory": "1024",
  "taskRoleArn": "{생성 Task-Role-IAM-Role-ARN}",
  "compatibilities": [
    "EC2",
    "FARGATE"
  ],
  "requiresCompatibilities": [
    "FARGATE"
  ],
  "networkMode": "awsvpc",
  "runtimePlatform": {
    "operatingSystemFamily": "LINUX",
    "cpuArchitecture": null
  },
  "cpu": "512",
}
```

# 4. 번외 : Slack Notification

Slack을 사용해서 Flow에 대한 알람을 받도록 만들 수가 있다. 그렇게 하기 위해선 따로 거쳐야 하는 과정이 있고 아래와 같다. 

Prefect의 Backend Server와 Agent의 ~/.prefect/config.toml 파일에 SLACK_WEB_HOOK_URL 을 추가한다.

기본값이 SLACK_WEB_HOOK_URL이고 다른 변수명을 사용해서 저장해서 다른 state_handler에서 사용할 때 변경해서 사용할 수 있다 

```bash
[server]
[server.ui]
apollo_url = "{Prefect Server Host}/graphql"

[cloud]
api = "{Prefect Server Host}:4200"

[context.secrets]
SLACK_WEBHOOK_URL = "{Slack WebHook URL}"
```

# 5. 참고 자료

[Prefect Workflow Automation with Azure DevOps and AKS | Infinite Lambda](https://infinitelambda.com/post/prefect-workflow-automation-azure-devops-aks/)

[Deploying Prefect Server with AWS ECS and Docker Storage](https://towardsdatascience.com/deploying-prefect-server-with-aws-ecs-fargate-and-docker-storage-36f633226c5f)

[Serverless Data Pipelines Made Easy with Prefect and AWS ECS Fargate](https://towardsdatascience.com/serverless-data-pipelines-made-easy-with-prefect-and-aws-ecs-fargate-7e25bacb450c#8815)

[How to Cut Your AWS ECS Costs with Fargate Spot and Prefect](https://towardsdatascience.com/how-to-cut-your-aws-ecs-costs-with-fargate-spot-and-prefect-1a1ba5d2e2df)

[Distributed Data Pipelines with AWS ECS Fargate and Prefect Cloud](https://www.lejimmy.com/distributed-data-pipelines-with-aws-ecs-fargate-and-prefect-cloud/)