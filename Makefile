tag := $(shell git symbolic-ref -q --short HEAD)

docker-build:
	docker build -t millionintegrals/vel:$(tag) .

docker-publish:
	docker push millionintegrals/vel:$(tag)

docker-run:
	docker run --rm -it millionintegrals/vel:$(tag)

docker-shell:
	docker run --rm -it millionintegrals/vel:$(tag) bash

docker-run-cuda:
	docker run --runtime=nvidia --rm -it millionintegrals/vel:$(tag)

docker-shell-cuda:
	docker run --runtime=nvidia --rm -it millionintegrals/vel:$(tag) bash

test:
	pytest .