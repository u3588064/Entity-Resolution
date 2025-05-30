# Use an official Node.js runtime as a parent image
FROM node:18-alpine

# Set the working directory in the container
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install production dependencies only
# Use --ignore-scripts for potentially improved security
RUN npm install

# Copy the compiled application code
COPY build ./build



# Define the command to run the application
CMD ["node", "dist/index.js"]
