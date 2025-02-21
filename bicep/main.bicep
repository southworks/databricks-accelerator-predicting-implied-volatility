@description('The name of the Azure Databricks workspace to create.')
param databricksResourceName string

var acceleratorRepoName = 'databricks-accelerator-predicting-implied-volatility'
var randomString = uniqueString(resourceGroup().id, databricksResourceName, acceleratorRepoName)
var managedResourceGroupName = 'databricks-rg-${databricksResourceName}-${randomString}'
var location = resourceGroup().location

// Managed Identity
resource managedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: randomString
  location: location
}

// Role Assignment (Contributor Role)
resource resourceGroupRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(randomString)
  scope: resourceGroup()
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      'b24988ac-6180-42a0-ab88-20f7382dd24c' // Contributor role ID
    )
    principalId: managedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource createDatabricks 'Microsoft.Resources/deploymentScripts@2023-08-01' = {
  name: 'create-or-update-databricks-${randomString}'
  location: location
  kind: 'AzurePowerShell'
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${managedIdentity.id}': {}
    }
  }
  properties: {
    azPowerShellVersion: '9.0'
    arguments: '-resourceName ${databricksResourceName} -resourceGroupName  ${resourceGroup().name} -location ${location} -sku premium -managedResourceGroupName ${managedResourceGroupName}'
    scriptContent: '''
      param([string] $resourceName,
        [string] $resourceGroupName,
        [string] $location,
        [string] $sku,
        [string] $managedResourceGroupName)
      # Check if workspace exists
      $resource = Get-AzDatabricksWorkspace -Name $resourceName -ResourceGroupName $resourceGroupName | Select-Object -Property ResourceId
      if (-not $resource) {
        # Create new workspace
        Write-Output "Creating new Databricks workspace: $resourceName"
        New-AzDatabricksWorkspace -Name $resourceName `
          -ResourceGroupName $resourceGroupName `
          -Location $location `
          -ManagedResourceGroupName $managedResourceGroupName `
          -Sku $sku
        # Wait for provisioning to complete
        $retryCount = 0
        do {
          Start-Sleep -Seconds 15
          $provisioningState = (Get-AzDatabricksWorkspace -Name $resourceName -ResourceGroupName $resourceGroupName).ProvisioningState
          Write-Output "Current state: $provisioningState (attempt $retryCount)"
          $retryCount++
        } while ($provisioningState -ne 'Succeeded' -and $retryCount -le 40)
      }
      # Output the workspace ID to signal completion
      $workspace = Get-AzDatabricksWorkspace -Name $resourceName -ResourceGroupName $resourceGroupName
      echo "{\"WorkspaceId\": \"$workspace.Id\", \"Exists\": \"True"}" > $AZ_SCRIPTS_OUTPUT_PATH
    '''
    timeout: 'PT1H'
    cleanupPreference: 'OnSuccess'
    retentionInterval: 'PT2H'
  }
}

module databricksModule './databricks.bicep' = {
  name: 'databricks-module-${randomString}'
  params: {
    acceleratorRepoName: acceleratorRepoName
    databricksResourceName: databricksResourceName
    location: location
    managedIdentityName: randomString
  }
  dependsOn: [
    createDatabricks
  ]
}

// Outputs
output databricksWorkspaceUrl string = databricksModule.outputs.databricksWorkspaceUrl
output databricksJobUrl string = databricksModule.outputs.databricksJobUrl
