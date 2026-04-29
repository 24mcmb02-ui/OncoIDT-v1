FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# Override API URL to point to the gateway inside Docker network
ENV VITE_API_BASE_URL=http://localhost:8000
ENV VITE_WS_BASE_URL=ws://localhost:8010

EXPOSE 3000

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3000", "--strictPort"]
