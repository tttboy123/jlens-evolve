FROM --platform=linux/amd64 php:8.3.16

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt update && apt install -y \
    wget \
    git \
    build-essential \
    libgd-dev \
    libzip-dev \
    libgmp-dev \
    libftp-dev \
    libcurl4-openssl-dev \
    && apt-get -y autoclean \
    && rm -rf /var/lib/apt/lists/*

# PHP's bundled libtool emits a raw `-soname` flag for the GMP extension under
# linux/amd64 emulation on Apple Silicon. GCC only accepts the linker option via
# `-Wl,-soname,<name>`. Configure GMP separately so the compatibility rewrite is
# narrow, deterministic, and leaves the remaining official extension set intact.
RUN docker-php-ext-install gd zip \
    && docker-php-source extract \
    && docker-php-ext-configure gmp \
    && sed -i 's/ \$soname / -Wl,-soname,\$soname /g' /usr/src/php/ext/gmp/libtool \
    && docker-php-ext-install -j1 gmp \
    && docker-php-ext-install ftp curl pcntl

RUN curl -sS https://getcomposer.org/installer | php -- --2.2 --install-dir=/usr/local/bin --filename=composer

RUN adduser --disabled-password --gecos 'dog' nonroot
