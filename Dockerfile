FROM ubuntu
COPY . /app
RUN apt-get update
RUN apt-get install -y nginx
RUN apt-get clean
RUN rm -rf /var/www
RUN mkdir -p var/www/mysite
RUN mkdir var/www/mysite/img
COPY index.html var/www/mysite
COPY image.png var/www/mysite/img
COPY index.html /usr/share/nginx/html/index.html
RUN chmod u+wrx,o-w var/www/mysite
RUN useradd SvetlanaFarbovskaya
RUN groupadd SvetlanyFarbovskie
RUN usermod -G SvetlanyFarbovskie SvetlanaFarbovskaya
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]