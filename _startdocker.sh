#!/bin/bash
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t frontrun . || exit 1
#docker build  -t frontrun . || exit 1

mkdir -p "$HOME/.gemini"
docker run -it --rm \
  -v "$(pwd):/app" \
  -v "$HOME/.gemini:/home/ubuntu/.gemini" \
  -v "$HOME/personal/dotfiles/.gitconfig:/home/ubuntu/.gitconfig:ro" \
  frontrun bash
