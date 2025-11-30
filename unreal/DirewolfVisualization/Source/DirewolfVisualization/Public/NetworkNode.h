// DIREWOLF Network Node Actor
// Represents a network device in 3D space

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/StaticMeshComponent.h"
#include "NiagaraComponent.h"
#include "NetworkNode.generated.h"

UENUM(BlueprintType)
enum class ENodeType : uint8
{
	Server UMETA(DisplayName = "Server"),
	Workstation UMETA(DisplayName = "Workstation"),
	Router UMETA(DisplayName = "Router"),
	Switch UMETA(DisplayName = "Switch"),
	Firewall UMETA(DisplayName = "Firewall"),
	Threat UMETA(DisplayName = "Threat"),
	Unknown UMETA(DisplayName = "Unknown")
};

UENUM(BlueprintType)
enum class EThreatLevel : uint8
{
	None UMETA(DisplayName = "None"),
	Low UMETA(DisplayName = "Low"),
	Medium UMETA(DisplayName = "Medium"),
	High UMETA(DisplayName = "High"),
	Critical UMETA(DisplayName = "Critical")
};

UCLASS()
class DIREWOLFVISUALIZATION_API ANetworkNode : public AActor
{
	GENERATED_BODY()
	
public:	
	ANetworkNode();

protected:
	virtual void BeginPlay() override;

public:	
	virtual void Tick(float DeltaTime) override;

	// Components
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
	UStaticMeshComponent* NodeMesh;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
	UNiagaraComponent* ThreatEffect;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
	UNiagaraComponent* PulseEffect;

	// Node Properties
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Node")
	FString NodeID;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Node")
	FString NodeName;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Node")
	FString IPAddress;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Node")
	ENodeType NodeType;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Node")
	EThreatLevel ThreatLevel;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Node")
	bool bIsHighlighted;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Node")
	bool bIsSelected;

	// Visual Properties
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Visual")
	FLinearColor NodeColor;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Visual")
	float NodeScale;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Visual")
	float PulseIntensity;

	// Functions
	UFUNCTION(BlueprintCallable, Category = "Node")
	void SetThreatLevel(EThreatLevel NewThreatLevel);

	UFUNCTION(BlueprintCallable, Category = "Node")
	void SetHighlighted(bool bHighlight);

	UFUNCTION(BlueprintCallable, Category = "Node")
	void SetSelected(bool bSelect);

	UFUNCTION(BlueprintCallable, Category = "Node")
	void StartThreatAnimation();

	UFUNCTION(BlueprintCallable, Category = "Node")
	void StopThreatAnimation();

	UFUNCTION(BlueprintCallable, Category = "Node")
	FLinearColor GetColorForNodeType(ENodeType Type);

	UFUNCTION(BlueprintCallable, Category = "Node")
	FLinearColor GetColorForThreatLevel(EThreatLevel Level);

	UFUNCTION(BlueprintCallable, Category = "Node")
	UStaticMesh* GetMeshForNodeType(ENodeType Type);

private:
	void UpdateVisuals();
	void UpdateThreatEffects();

	float PulseTime;
	bool bThreatAnimationActive;
};
