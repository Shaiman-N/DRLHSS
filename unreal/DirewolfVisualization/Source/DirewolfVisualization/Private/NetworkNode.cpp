// DIREWOLF Network Node Implementation

#include "NetworkNode.h"
#include "NiagaraFunctionLibrary.h"
#include "Materials/MaterialInstanceDynamic.h"

ANetworkNode::ANetworkNode()
{
	PrimaryActorTick.bCanEverTick = true;

	// Create root component
	RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));

	// Create mesh component
	NodeMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("NodeMesh"));
	NodeMesh->SetupAttachment(RootComponent);
	NodeMesh->SetCollisionEnabled(ECollisionEnabled::QueryOnly);
	NodeMesh->SetCollisionResponseToAllChannels(ECR_Block);

	// Create Niagara components
	ThreatEffect = CreateDefaultSubobject<UNiagaraComponent>(TEXT("ThreatEffect"));
	ThreatEffect->SetupAttachment(NodeMesh);
	ThreatEffect->bAutoActivate = false;

	PulseEffect = CreateDefaultSubobject<UNiagaraComponent>(TEXT("PulseEffect"));
	PulseEffect->SetupAttachment(NodeMesh);
	PulseEffect->bAutoActivate = false;

	// Initialize properties
	NodeType = ENodeType::Unknown;
	ThreatLevel = EThreatLevel::None;
	bIsHighlighted = false;
	bIsSelected = false;
	NodeScale = 1.0f;
	PulseIntensity = 0.0f;
	PulseTime = 0.0f;
	bThreatAnimationActive = false;
}

void ANetworkNode::BeginPlay()
{
	Super::BeginPlay();
	
	// Set initial mesh based on node type
	UStaticMesh* Mesh = GetMeshForNodeType(NodeType);
	if (Mesh)
	{
		NodeMesh->SetStaticMesh(Mesh);
	}

	// Set initial color
	NodeColor = GetColorForNodeType(NodeType);
	UpdateVisuals();
}

void ANetworkNode::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	if (bThreatAnimationActive)
	{
		PulseTime += DeltaTime;
		PulseIntensity = 0.5f + 0.5f * FMath::Sin(PulseTime * 3.0f);
		UpdateThreatEffects();
	}
}

void ANetworkNode::SetThreatLevel(EThreatLevel NewThreatLevel)
{
	ThreatLevel = NewThreatLevel;
	
	// Update color based on threat level
	if (ThreatLevel != EThreatLevel::None)
	{
		NodeColor = GetColorForThreatLevel(ThreatLevel);
	}
	else
	{
		NodeColor = GetColorForNodeType(NodeType);
	}

	UpdateVisuals();

	// Start threat animation for high/critical threats
	if (ThreatLevel >= EThreatLevel::High)
	{
		StartThreatAnimation();
	}
	else
	{
		StopThreatAnimation();
	}
}

void ANetworkNode::SetHighlighted(bool bHighlight)
{
	bIsHighlighted = bHighlight;
	UpdateVisuals();
}

void ANetworkNode::SetSelected(bool bSelect)
{
	bIsSelected = bSelect;
	UpdateVisuals();
}

void ANetworkNode::StartThreatAnimation()
{
	bThreatAnimationActive = true;
	PulseTime = 0.0f;

	if (ThreatEffect)
	{
		ThreatEffect->Activate();
	}

	if (PulseEffect)
	{
		PulseEffect->Activate();
	}
}

void ANetworkNode::StopThreatAnimation()
{
	bThreatAnimationActive = false;

	if (ThreatEffect)
	{
		ThreatEffect->Deactivate();
	}

	if (PulseEffect)
	{
		PulseEffect->Deactivate();
	}
}

FLinearColor ANetworkNode::GetColorForNodeType(ENodeType Type)
{
	switch (Type)
	{
	case ENodeType::Server:
		return FLinearColor(0.3f, 0.6f, 1.0f); // Blue
	case ENodeType::Workstation:
		return FLinearColor(0.5f, 0.8f, 0.5f); // Green
	case ENodeType::Router:
		return FLinearColor(0.8f, 0.6f, 0.3f); // Orange
	case ENodeType::Switch:
		return FLinearColor(0.7f, 0.7f, 0.3f); // Yellow
	case ENodeType::Firewall:
		return FLinearColor(0.9f, 0.5f, 0.2f); // Dark Orange
	case ENodeType::Threat:
		return FLinearColor(1.0f, 0.2f, 0.2f); // Red
	default:
		return FLinearColor(0.5f, 0.5f, 0.5f); // Gray
	}
}

FLinearColor ANetworkNode::GetColorForThreatLevel(EThreatLevel Level)
{
	switch (Level)
	{
	case EThreatLevel::Low:
		return FLinearColor(1.0f, 1.0f, 0.3f); // Yellow
	case EThreatLevel::Medium:
		return FLinearColor(1.0f, 0.6f, 0.2f); // Orange
	case EThreatLevel::High:
		return FLinearColor(1.0f, 0.3f, 0.2f); // Red-Orange
	case EThreatLevel::Critical:
		return FLinearColor(1.0f, 0.0f, 0.0f); // Red
	default:
		return GetColorForNodeType(NodeType);
	}
}

UStaticMesh* ANetworkNode::GetMeshForNodeType(ENodeType Type)
{
	// Load appropriate mesh based on node type
	// In production, these would be actual asset references
	FString MeshPath;

	switch (Type)
	{
	case ENodeType::Server:
		MeshPath = TEXT("/Game/Meshes/Server_Mesh.Server_Mesh");
		break;
	case ENodeType::Workstation:
		MeshPath = TEXT("/Game/Meshes/Workstation_Mesh.Workstation_Mesh");
		break;
	case ENodeType::Router:
		MeshPath = TEXT("/Game/Meshes/Router_Mesh.Router_Mesh");
		break;
	case ENodeType::Switch:
		MeshPath = TEXT("/Game/Meshes/Switch_Mesh.Switch_Mesh");
		break;
	case ENodeType::Firewall:
		MeshPath = TEXT("/Game/Meshes/Firewall_Mesh.Firewall_Mesh");
		break;
	case ENodeType::Threat:
		MeshPath = TEXT("/Game/Meshes/Threat_Mesh.Threat_Mesh");
		break;
	default:
		MeshPath = TEXT("/Engine/BasicShapes/Cube.Cube");
	}

	return LoadObject<UStaticMesh>(nullptr, *MeshPath);
}

void ANetworkNode::UpdateVisuals()
{
	if (!NodeMesh)
		return;

	// Create dynamic material instance
	UMaterialInstanceDynamic* DynMaterial = NodeMesh->CreateDynamicMaterialInstance(0);
	if (DynMaterial)
	{
		// Set base color
		DynMaterial->SetVectorParameterValue(TEXT("BaseColor"), NodeColor);

		// Set emissive for highlighted/selected
		if (bIsSelected)
		{
			DynMaterial->SetVectorParameterValue(TEXT("EmissiveColor"), FLinearColor(1.0f, 1.0f, 0.0f));
			DynMaterial->SetScalarParameterValue(TEXT("EmissiveIntensity"), 5.0f);
		}
		else if (bIsHighlighted)
		{
			DynMaterial->SetVectorParameterValue(TEXT("EmissiveColor"), FLinearColor(0.5f, 0.5f, 1.0f));
			DynMaterial->SetScalarParameterValue(TEXT("EmissiveIntensity"), 2.0f);
		}
		else
		{
			DynMaterial->SetScalarParameterValue(TEXT("EmissiveIntensity"), 0.0f);
		}
	}

	// Update scale
	NodeMesh->SetRelativeScale3D(FVector(NodeScale));
}

void ANetworkNode::UpdateThreatEffects()
{
	if (ThreatEffect)
	{
		ThreatEffect->SetFloatParameter(TEXT("Intensity"), PulseIntensity);
		ThreatEffect->SetColorParameter(TEXT("Color"), NodeColor);
	}

	if (PulseEffect)
	{
		PulseEffect->SetFloatParameter(TEXT("PulseScale"), 1.0f + PulseIntensity * 0.5f);
	}
}
