FROM mcr.microsoft.com/mssql/server:2022-latest
USER root
# Still root
RUN apt-get update \
 && apt-get install nano
# Copy in application (still as root, won't be writable by other users)
COPY data/vPICList_lite_2023_06.bak vPICList_lite_2023_06.bak
# Now as the last step create a user and default to running as it
RUN adduser ros
USER ros