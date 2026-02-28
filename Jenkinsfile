pipeline {
    agent any

    environment {
        DOCKER_CREDS = credentials('docker-hub-credentials')
        DOCKER_USER = "2022bcs0088shashank"
        ROLL_NO = "2022bcs0088"
        NAME = "Shashank Upadhyay"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Train and Evaluate') {
            steps {
                script {
                    sh "pip3 install -r requirements.txt --break-system-packages"
                    sh "python3 train.py" 
                    
                    
                    def metrics = readJSON file: 'output/results.json'
                    echo "Student Name: ${NAME}"
                    echo "Roll Number: ${ROLL_NO}"
                    echo "MSE: ${metrics.mse}"
                    echo "R2: ${metrics.r2}"
                }
            }
        }

        stage('Deploy to Docker Hub') {
            steps {
                script {
                    
                    docker.withRegistry('https://index.docker.io/v1/', 'docker-hub-credentials') {
                    def appImage = docker.build("${DOCKER_USER}/lab6:latest")
                    appImage.push()
                    }
                }
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'output/**', fingerprint: true
        }
    }
}