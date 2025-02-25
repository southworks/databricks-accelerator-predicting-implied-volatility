param acceleratorRepoName string
param databricksResourceName string
param location string
param managedIdentityName string

resource databricks 'Microsoft.Databricks/workspaces@2024-05-01' existing = {
  name: databricksResourceName
}

resource managedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' existing = {
  name: managedIdentityName
}

// Deployment Script
resource jobCreation 'Microsoft.Resources/deploymentScripts@2023-08-01' = {
  name: 'job-creation-${acceleratorRepoName}'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${managedIdentity.id}': {}
    }
  }
  kind: 'AzureCLI'
  properties: {
    azCliVersion: '2.9.1'
    scriptContent: '''
      set -e

      # Install Databricks CLI
      echo "Installing Databricks CLI..."
      curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh

      # Wait for Azure Databricks resource to finish creating the MRG storage containers
      echo "Testing connection and waiting for storage initialization..."
      max_attempts=30
      attempt=0
      while [ $attempt -lt $max_attempts ]; do
        if databricks fs ls dbfs:/; then
          echo "Storage initialized successfully"
          break
        fi
        echo "Waiting for storage initialization... (attempt $((attempt + 1)))"
        sleep 10
        attempt=$((attempt + 1))
      done
      if [ $attempt -eq $max_attempts ]; then
        echo "Timeout waiting for storage initialization"
        exit 1
      fi

      # Check if the repo exists; if not, create it. If it exists, update it
      repo_path="/Users/${ARM_CLIENT_ID}/${ACCELERATOR_REPO_NAME}"
      repo_info=$(databricks repos get "${repo_path}" 2>/dev/null || true)

      if [ -z "$repo_info" ]; then
          echo "Repository does not exist. Creating..."
          databricks repos create https://github.com/southworks/${ACCELERATOR_REPO_NAME} gitHub
          repo_id=$(databricks repos get "${repo_path}" | jq -r '.id')
          databricks repos update ${repo_id} --branch ${BRANCH_NAME}
      else
          echo "Repository exists. Updating to latest main branch..."
          repo_id=$(databricks repos get "${repo_path}" | jq -r '.id')
          databricks repos update ${repo_id} --branch ${BRANCH_NAME}
      fi

      # Add RUNME.py path into job-template.json
      databricks workspace export ${repo_path}/bicep/job-template.json > job-template.json
      notebook_path="${repo_path}/RUNME"
      jq ".tasks[0].notebook_task.notebook_path = \"${notebook_path}\"" job-template.json > job.json

      # Create and run Databricks job
      job_page_url=$(databricks jobs submit --json @./job.json | jq -r '.run_page_url')
      echo "{\"job_page_url\": \"$job_page_url\"}" > $AZ_SCRIPTS_OUTPUT_PATH
      '''
    environmentVariables: [
      {
        name: 'DATABRICKS_AZURE_RESOURCE_ID'
        value: databricks.id
      }
      {
        name: 'ARM_CLIENT_ID'
        value: managedIdentity.properties.clientId
      }
      {
        name: 'ARM_USE_MSI'
        value: 'true'
      }
      {
        name: 'ACCELERATOR_REPO_NAME'
        value: acceleratorRepoName
      }
      {
        name: 'BRANCH_NAME'
        value: 'main'
      }
    ]
    timeout: 'PT1H'
    cleanupPreference: 'OnSuccess'
    retentionInterval: 'PT2H'
  }
}

output databricksWorkspaceUrl string = 'https://${databricks.properties.workspaceUrl}'
output databricksJobUrl string = jobCreation.properties.outputs.job_page_url
