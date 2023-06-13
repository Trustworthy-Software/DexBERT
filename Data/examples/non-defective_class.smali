.class final Landroid/support/v4/app/BackStackState;
.super Ljava/lang/Object;
.source "BackStackRecord.java"

# interfaces
.implements Landroid/os/Parcelable;


# static fields
.field public static final CREATOR:Landroid/os/Parcelable$Creator;
    .annotation system Ldalvik/annotation/Signature;
        value = {
            "Landroid/os/Parcelable$Creator",
            "<",
            "Landroid/support/v4/app/BackStackState;",
            ">;"
        }
    .end annotation
.end field


# instance fields
.field final mBreadCrumbShortTitleRes:I

.field final mBreadCrumbShortTitleText:Ljava/lang/CharSequence;

.field final mBreadCrumbTitleRes:I

.field final mBreadCrumbTitleText:Ljava/lang/CharSequence;

.field final mIndex:I

.field final mName:Ljava/lang/String;

.field final mOps:[I

.field final mTransition:I

.field final mTransitionStyle:I


# direct methods
.method static constructor <clinit>()V
    .registers 1

    .prologue
    .line 149
    new-instance v0, Landroid/support/v4/app/BackStackState$1;

    invoke-direct {v0}, Landroid/support/v4/app/BackStackState$1;-><init>()V

    sput-object v0, Landroid/support/v4/app/BackStackState;->CREATOR:Landroid/os/Parcelable$Creator;

    return-void
.end method

.method public constructor <init>(Landroid/os/Parcel;)V
    .registers 3
    .param p1, "in"    # Landroid/os/Parcel;

    .prologue
    .line 82
    invoke-direct {p0}, Ljava/lang/Object;-><init>()V

    .line 83
    invoke-virtual {p1}, Landroid/os/Parcel;->createIntArray()[I

    move-result-object v0

    iput-object v0, p0, Landroid/support/v4/app/BackStackState;->mOps:[I

    .line 84
    invoke-virtual {p1}, Landroid/os/Parcel;->readInt()I

    move-result v0

    iput v0, p0, Landroid/support/v4/app/BackStackState;->mTransition:I

    .line 85
    invoke-virtual {p1}, Landroid/os/Parcel;->readInt()I

    move-result v0

    iput v0, p0, Landroid/support/v4/app/BackStackState;->mTransitionStyle:I

    .line 86
    invoke-virtual {p1}, Landroid/os/Parcel;->readString()Ljava/lang/String;

    move-result-object v0

    iput-object v0, p0, Landroid/support/v4/app/BackStackState;->mName:Ljava/lang/String;

    .line 87
    invoke-virtual {p1}, Landroid/os/Parcel;->readInt()I

    move-result v0

    iput v0, p0, Landroid/support/v4/app/BackStackState;->mIndex:I

    .line 88
    invoke-virtual {p1}, Landroid/os/Parcel;->readInt()I

    move-result v0

    iput v0, p0, Landroid/support/v4/app/BackStackState;->mBreadCrumbTitleRes:I

    .line 89
    sget-object v0, Landroid/text/TextUtils;->CHAR_SEQUENCE_CREATOR:Landroid/os/Parcelable$Creator;

    invoke-interface {v0, p1}, Landroid/os/Parcelable$Creator;->createFromParcel(Landroid/os/Parcel;)Ljava/lang/Object;

    move-result-object v0

    check-cast v0, Ljava/lang/CharSequence;

    iput-object v0, p0, Landroid/support/v4/app/BackStackState;->mBreadCrumbTitleText:Ljava/lang/CharSequence;

    .line 90
    invoke-virtual {p1}, Landroid/os/Parcel;->readInt()I

    move-result v0

    iput v0, p0, Landroid/support/v4/app/BackStackState;->mBreadCrumbShortTitleRes:I

    .line 91
    sget-object v0, Landroid/text/TextUtils;->CHAR_SEQUENCE_CREATOR:Landroid/os/Parcelable$Creator;

    invoke-interface {v0, p1}, Landroid/os/Parcelable$Creator;->createFromParcel(Landroid/os/Parcel;)Ljava/lang/Object;

    move-result-object v0

    check-cast v0, Ljava/lang/CharSequence;

    iput-object v0, p0, Landroid/support/v4/app/BackStackState;->mBreadCrumbShortTitleText:Ljava/lang/CharSequence;

    .line 92
    return-void
.end method

.method public constructor <init>(Landroid/support/v4/app/FragmentManagerImpl;Landroid/support/v4/app/BackStackRecord;)V
    .registers 11
    .param p1, "fm"    # Landroid/support/v4/app/FragmentManagerImpl;
    .param p2, "bse"    # Landroid/support/v4/app/BackStackRecord;

    .prologue
    .line 39
    invoke-direct {p0}, Ljava/lang/Object;-><init>()V

    .line 40
    const/4 v2, 0x0

    .line 41
    .local v2, "numRemoved":I
    iget-object v3, p2, Landroid/support/v4/app/BackStackRecord;->mHead:Landroid/support/v4/app/BackStackRecord$Op;

    .line 42
    .local v3, "op":Landroid/support/v4/app/BackStackRecord$Op;
    :goto_6
    if-eqz v3, :cond_16

    .line 43
    iget-object v6, v3, Landroid/support/v4/app/BackStackRecord$Op;->removed:Ljava/util/ArrayList;

    if-eqz v6, :cond_13

    iget-object v6, v3, Landroid/support/v4/app/BackStackRecord$Op;->removed:Ljava/util/ArrayList;

    invoke-virtual {v6}, Ljava/util/ArrayList;->size()I

    move-result v6

    add-int/2addr v2, v6

    .line 44
    :cond_13
    iget-object v3, v3, Landroid/support/v4/app/BackStackRecord$Op;->next:Landroid/support/v4/app/BackStackRecord$Op;

    goto :goto_6

    .line 46
    :cond_16
    iget v6, p2, Landroid/support/v4/app/BackStackRecord;->mNumOp:I

    mul-int/lit8 v6, v6, 0x7

    add-int/2addr v6, v2

    new-array v6, v6, [I

    iput-object v6, p0, Landroid/support/v4/app/BackStackState;->mOps:[I

    .line 48
    iget-boolean v6, p2, Landroid/support/v4/app/BackStackRecord;->mAddToBackStack:Z

    if-nez v6, :cond_2b

    .line 49
    new-instance v6, Ljava/lang/IllegalStateException;

    const-string v7, "Not on back stack"

    invoke-direct {v6, v7}, Ljava/lang/IllegalStateException;-><init>(Ljava/lang/String;)V

    throw v6

    .line 52
    :cond_2b
    iget-object v3, p2, Landroid/support/v4/app/BackStackRecord;->mHead:Landroid/support/v4/app/BackStackRecord$Op;

    .line 53
    const/4 v4, 0x0

    .local v4, "pos":I
    move v5, v4

    .line 54
    .end local v4    # "pos":I
    .local v5, "pos":I
    :goto_2f
    if-eqz v3, :cond_98

    .line 55
    iget-object v6, p0, Landroid/support/v4/app/BackStackState;->mOps:[I

    add-int/lit8 v4, v5, 0x1

    .end local v5    # "pos":I
    .restart local v4    # "pos":I
    iget v7, v3, Landroid/support/v4/app/BackStackRecord$Op;->cmd:I

    aput v7, v6, v5

    .line 56
    iget-object v6, p0, Landroid/support/v4/app/BackStackState;->mOps:[I

    add-int/lit8 v5, v4, 0x1

    .end local v4    # "pos":I
    .restart local v5    # "pos":I
    iget-object v7, v3, Landroid/support/v4/app/BackStackRecord$Op;->fragment:Landroid/support/v4/app/Fragment;

    iget v7, v7, Landroid/support/v4/app/Fragment;->mIndex:I

    aput v7, v6, v4

    .line 57
    iget-object v6, p0, Landroid/support/v4/app/BackStackState;->mOps:[I

    add-int/lit8 v4, v5, 0x1

    .end local v5    # "pos":I
    .restart local v4    # "pos":I
    iget v7, v3, Landroid/support/v4/app/BackStackRecord$Op;->enterAnim:I

    aput v7, v6, v5

    .line 58
    iget-object v6, p0, Landroid/support/v4/app/BackStackState;->mOps:[I

    add-int/lit8 v5, v4, 0x1

    .end local v4    # "pos":I
    .restart local v5    # "pos":I
    iget v7, v3, Landroid/support/v4/app/BackStackRecord$Op;->exitAnim:I

    aput v7, v6, v4

    .line 59
    iget-object v6, p0, Landroid/support/v4/app/BackStackState;->mOps:[I

    add-int/lit8 v4, v5, 0x1

    .end local v5    # "pos":I
    .restart local v4    # "pos":I
    iget v7, v3, Landroid/support/v4/app/BackStackRecord$Op;->popEnterAnim:I

    aput v7, v6, v5

    .line 60
    iget-object v6, p0, Landroid/support/v4/app/BackStackState;->mOps:[I

    add-int/lit8 v5, v4, 0x1

    .end local v4    # "pos":I
    .restart local v5    # "pos":I
    iget v7, v3, Landroid/support/v4/app/BackStackRecord$Op;->popExitAnim:I

    aput v7, v6, v4

    .line 61
    iget-object v6, v3, Landroid/support/v4/app/BackStackRecord$Op;->removed:Ljava/util/ArrayList;

    if-eqz v6, :cond_90

    .line 62
    iget-object v6, v3, Landroid/support/v4/app/BackStackRecord$Op;->removed:Ljava/util/ArrayList;

    invoke-virtual {v6}, Ljava/util/ArrayList;->size()I

    move-result v0

    .line 63
    .local v0, "N":I
    iget-object v6, p0, Landroid/support/v4/app/BackStackState;->mOps:[I

    add-int/lit8 v4, v5, 0x1

    .end local v5    # "pos":I
    .restart local v4    # "pos":I
    aput v0, v6, v5

    .line 64
    const/4 v1, 0x0

    .local v1, "i":I
    move v5, v4

    .end local v4    # "pos":I
    .restart local v5    # "pos":I
    :goto_75
    if-ge v1, v0, :cond_8b

    .line 65
    iget-object v7, p0, Landroid/support/v4/app/BackStackState;->mOps:[I

    add-int/lit8 v4, v5, 0x1

    .end local v5    # "pos":I
    .restart local v4    # "pos":I
    iget-object v6, v3, Landroid/support/v4/app/BackStackRecord$Op;->removed:Ljava/util/ArrayList;

    invoke-virtual {v6, v1}, Ljava/util/ArrayList;->get(I)Ljava/lang/Object;

    move-result-object v6

    check-cast v6, Landroid/support/v4/app/Fragment;

    iget v6, v6, Landroid/support/v4/app/Fragment;->mIndex:I

    aput v6, v7, v5

    .line 64
    add-int/lit8 v1, v1, 0x1

    move v5, v4

    .end local v4    # "pos":I
    .restart local v5    # "pos":I
    goto :goto_75

    :cond_8b
    move v4, v5

    .line 70
    .end local v0    # "N":I
    .end local v1    # "i":I
    .end local v5    # "pos":I
    .restart local v4    # "pos":I
    :goto_8c
    iget-object v3, v3, Landroid/support/v4/app/BackStackRecord$Op;->next:Landroid/support/v4/app/BackStackRecord$Op;

    move v5, v4

    .end local v4    # "pos":I
    .restart local v5    # "pos":I
    goto :goto_2f

    .line 68
    :cond_90
    iget-object v6, p0, Landroid/support/v4/app/BackStackState;->mOps:[I

    add-int/lit8 v4, v5, 0x1

    .end local v5    # "pos":I
    .restart local v4    # "pos":I
    const/4 v7, 0x0

    aput v7, v6, v5

    goto :goto_8c

    .line 72
    .end local v4    # "pos":I
    .restart local v5    # "pos":I
    :cond_98
    iget v6, p2, Landroid/support/v4/app/BackStackRecord;->mTransition:I

    iput v6, p0, Landroid/support/v4/app/BackStackState;->mTransition:I

    .line 73
    iget v6, p2, Landroid/support/v4/app/BackStackRecord;->mTransitionStyle:I

    iput v6, p0, Landroid/support/v4/app/BackStackState;->mTransitionStyle:I

    .line 74
    iget-object v6, p2, Landroid/support/v4/app/BackStackRecord;->mName:Ljava/lang/String;

    iput-object v6, p0, Landroid/support/v4/app/BackStackState;->mName:Ljava/lang/String;

    .line 75
    iget v6, p2, Landroid/support/v4/app/BackStackRecord;->mIndex:I

    iput v6, p0, Landroid/support/v4/app/BackStackState;->mIndex:I

    .line 76
    iget v6, p2, Landroid/support/v4/app/BackStackRecord;->mBreadCrumbTitleRes:I

    iput v6, p0, Landroid/support/v4/app/BackStackState;->mBreadCrumbTitleRes:I

    .line 77
    iget-object v6, p2, Landroid/support/v4/app/BackStackRecord;->mBreadCrumbTitleText:Ljava/lang/CharSequence;

    iput-object v6, p0, Landroid/support/v4/app/BackStackState;->mBreadCrumbTitleText:Ljava/lang/CharSequence;

    .line 78
    iget v6, p2, Landroid/support/v4/app/BackStackRecord;->mBreadCrumbShortTitleRes:I

    iput v6, p0, Landroid/support/v4/app/BackStackState;->mBreadCrumbShortTitleRes:I

    .line 79
    iget-object v6, p2, Landroid/support/v4/app/BackStackRecord;->mBreadCrumbShortTitleText:Ljava/lang/CharSequence;

    iput-object v6, p0, Landroid/support/v4/app/BackStackState;->mBreadCrumbShortTitleText:Ljava/lang/CharSequence;

    .line 80
    return-void
.end method


# virtual methods
.method public describeContents()I
    .registers 2

    .prologue
    .line 134
    const/4 v0, 0x0

    return v0
.end method

.method public instantiate(Landroid/support/v4/app/FragmentManagerImpl;)Landroid/support/v4/app/BackStackRecord;
    .registers 14
    .param p1, "fm"    # Landroid/support/v4/app/FragmentManagerImpl;

    .prologue
    const/4 v11, 0x1

    .line 95
    new-instance v1, Landroid/support/v4/app/BackStackRecord;

    invoke-direct {v1, p1}, Landroid/support/v4/app/BackStackRecord;-><init>(Landroid/support/v4/app/FragmentManagerImpl;)V

    .line 96
    .local v1, "bse":Landroid/support/v4/app/BackStackRecord;
    const/4 v5, 0x0

    .line 97
    .local v5, "pos":I
    :goto_7
    iget-object v8, p0, Landroid/support/v4/app/BackStackState;->mOps:[I

    array-length v8, v8

    if-ge v5, v8, :cond_cc

    .line 98
    new-instance v4, Landroid/support/v4/app/BackStackRecord$Op;

    invoke-direct {v4}, Landroid/support/v4/app/BackStackRecord$Op;-><init>()V

    .line 99
    .local v4, "op":Landroid/support/v4/app/BackStackRecord$Op;
    iget-object v8, p0, Landroid/support/v4/app/BackStackState;->mOps:[I

    add-int/lit8 v6, v5, 0x1

    .end local v5    # "pos":I
    .local v6, "pos":I
    aget v8, v8, v5

    iput v8, v4, Landroid/support/v4/app/BackStackRecord$Op;->cmd:I

    .line 100
    sget-boolean v8, Landroid/support/v4/app/FragmentManagerImpl;->DEBUG:Z

    if-eqz v8, :cond_43

    const-string v8, "FragmentManager"

    new-instance v9, Ljava/lang/StringBuilder;

    invoke-direct {v9}, Ljava/lang/StringBuilder;-><init>()V

    const-string v10, "BSE "

    invoke-virtual {v9, v10}, Ljava/lang/StringBuilder;->append(Ljava/lang/String;)Ljava/lang/StringBuilder;

    move-result-object v9

    invoke-virtual {v9, v1}, Ljava/lang/StringBuilder;->append(Ljava/lang/Object;)Ljava/lang/StringBuilder;

    move-result-object v9

    const-string v10, " set base fragment #"

    invoke-virtual {v9, v10}, Ljava/lang/StringBuilder;->append(Ljava/lang/String;)Ljava/lang/StringBuilder;

    move-result-object v9

    iget-object v10, p0, Landroid/support/v4/app/BackStackState;->mOps:[I

    aget v10, v10, v6

    invoke-virtual {v9, v10}, Ljava/lang/StringBuilder;->append(I)Ljava/lang/StringBuilder;

    move-result-object v9

    invoke-virtual {v9}, Ljava/lang/StringBuilder;->toString()Ljava/lang/String;

    move-result-object v9

    invoke-static {v8, v9}, Landroid/util/Log;->v(Ljava/lang/String;Ljava/lang/String;)I

    .line 102
    :cond_43
    iget-object v8, p1, Landroid/support/v4/app/FragmentManagerImpl;->mActive:Ljava/util/ArrayList;

    iget-object v9, p0, Landroid/support/v4/app/BackStackState;->mOps:[I

    add-int/lit8 v5, v6, 0x1

    .end local v6    # "pos":I
    .restart local v5    # "pos":I
    aget v9, v9, v6

    invoke-virtual {v8, v9}, Ljava/util/ArrayList;->get(I)Ljava/lang/Object;

    move-result-object v2

    check-cast v2, Landroid/support/v4/app/Fragment;

    .line 103
    .local v2, "f":Landroid/support/v4/app/Fragment;
    iput-object v2, v4, Landroid/support/v4/app/BackStackRecord$Op;->fragment:Landroid/support/v4/app/Fragment;

    .line 104
    iget-object v8, p0, Landroid/support/v4/app/BackStackState;->mOps:[I

    add-int/lit8 v6, v5, 0x1

    .end local v5    # "pos":I
    .restart local v6    # "pos":I
    aget v8, v8, v5

    iput v8, v4, Landroid/support/v4/app/BackStackRecord$Op;->enterAnim:I

    .line 105
    iget-object v8, p0, Landroid/support/v4/app/BackStackState;->mOps:[I

    add-int/lit8 v5, v6, 0x1

    .end local v6    # "pos":I
    .restart local v5    # "pos":I
    aget v8, v8, v6

    iput v8, v4, Landroid/support/v4/app/BackStackRecord$Op;->exitAnim:I

    .line 106
    iget-object v8, p0, Landroid/support/v4/app/BackStackState;->mOps:[I

    add-int/lit8 v6, v5, 0x1

    .end local v5    # "pos":I
    .restart local v6    # "pos":I
    aget v8, v8, v5

    iput v8, v4, Landroid/support/v4/app/BackStackRecord$Op;->popEnterAnim:I

    .line 107
    iget-object v8, p0, Landroid/support/v4/app/BackStackState;->mOps:[I

    add-int/lit8 v5, v6, 0x1

    .end local v6    # "pos":I
    .restart local v5    # "pos":I
    aget v8, v8, v6

    iput v8, v4, Landroid/support/v4/app/BackStackRecord$Op;->popExitAnim:I

    .line 108
    iget-object v8, p0, Landroid/support/v4/app/BackStackState;->mOps:[I

    add-int/lit8 v6, v5, 0x1

    .end local v5    # "pos":I
    .restart local v6    # "pos":I
    aget v0, v8, v5

    .line 109
    .local v0, "N":I
    if-lez v0, :cond_c6

    .line 110
    new-instance v8, Ljava/util/ArrayList;

    invoke-direct {v8, v0}, Ljava/util/ArrayList;-><init>(I)V

    iput-object v8, v4, Landroid/support/v4/app/BackStackRecord$Op;->removed:Ljava/util/ArrayList;

    .line 111
    const/4 v3, 0x0

    .local v3, "i":I
    :goto_83
    if-ge v3, v0, :cond_c6

    .line 112
    sget-boolean v8, Landroid/support/v4/app/FragmentManagerImpl;->DEBUG:Z

    if-eqz v8, :cond_af

    const-string v8, "FragmentManager"

    new-instance v9, Ljava/lang/StringBuilder;

    invoke-direct {v9}, Ljava/lang/StringBuilder;-><init>()V

    const-string v10, "BSE "

    invoke-virtual {v9, v10}, Ljava/lang/StringBuilder;->append(Ljava/lang/String;)Ljava/lang/StringBuilder;

    move-result-object v9

    invoke-virtual {v9, v1}, Ljava/lang/StringBuilder;->append(Ljava/lang/Object;)Ljava/lang/StringBuilder;

    move-result-object v9

    const-string v10, " set remove fragment #"

    invoke-virtual {v9, v10}, Ljava/lang/StringBuilder;->append(Ljava/lang/String;)Ljava/lang/StringBuilder;

    move-result-object v9

    iget-object v10, p0, Landroid/support/v4/app/BackStackState;->mOps:[I

    aget v10, v10, v6

    invoke-virtual {v9, v10}, Ljava/lang/StringBuilder;->append(I)Ljava/lang/StringBuilder;

    move-result-object v9

    invoke-virtual {v9}, Ljava/lang/StringBuilder;->toString()Ljava/lang/String;

    move-result-object v9

    invoke-static {v8, v9}, Landroid/util/Log;->v(Ljava/lang/String;Ljava/lang/String;)I

    .line 114
    :cond_af
    iget-object v8, p1, Landroid/support/v4/app/FragmentManagerImpl;->mActive:Ljava/util/ArrayList;

    iget-object v9, p0, Landroid/support/v4/app/BackStackState;->mOps:[I

    add-int/lit8 v5, v6, 0x1

    .end local v6    # "pos":I
    .restart local v5    # "pos":I
    aget v9, v9, v6

    invoke-virtual {v8, v9}, Ljava/util/ArrayList;->get(I)Ljava/lang/Object;

    move-result-object v7

    check-cast v7, Landroid/support/v4/app/Fragment;

    .line 115
    .local v7, "r":Landroid/support/v4/app/Fragment;
    iget-object v8, v4, Landroid/support/v4/app/BackStackRecord$Op;->removed:Ljava/util/ArrayList;

    invoke-virtual {v8, v7}, Ljava/util/ArrayList;->add(Ljava/lang/Object;)Z

    .line 111
    add-int/lit8 v3, v3, 0x1

    move v6, v5

    .end local v5    # "pos":I
    .restart local v6    # "pos":I
    goto :goto_83

    .end local v3    # "i":I
    .end local v7    # "r":Landroid/support/v4/app/Fragment;
    :cond_c6
    move v5, v6

    .line 118
    .end local v6    # "pos":I
    .restart local v5    # "pos":I
    invoke-virtual {v1, v4}, Landroid/support/v4/app/BackStackRecord;->addOp(Landroid/support/v4/app/BackStackRecord$Op;)V

    goto/16 :goto_7

    .line 120
    .end local v0    # "N":I
    .end local v2    # "f":Landroid/support/v4/app/Fragment;
    .end local v4    # "op":Landroid/support/v4/app/BackStackRecord$Op;
    :cond_cc
    iget v8, p0, Landroid/support/v4/app/BackStackState;->mTransition:I

    iput v8, v1, Landroid/support/v4/app/BackStackRecord;->mTransition:I

    .line 121
    iget v8, p0, Landroid/support/v4/app/BackStackState;->mTransitionStyle:I

    iput v8, v1, Landroid/support/v4/app/BackStackRecord;->mTransitionStyle:I

    .line 122
    iget-object v8, p0, Landroid/support/v4/app/BackStackState;->mName:Ljava/lang/String;

    iput-object v8, v1, Landroid/support/v4/app/BackStackRecord;->mName:Ljava/lang/String;

    .line 123
    iget v8, p0, Landroid/support/v4/app/BackStackState;->mIndex:I

    iput v8, v1, Landroid/support/v4/app/BackStackRecord;->mIndex:I

    .line 124
    iput-boolean v11, v1, Landroid/support/v4/app/BackStackRecord;->mAddToBackStack:Z

    .line 125
    iget v8, p0, Landroid/support/v4/app/BackStackState;->mBreadCrumbTitleRes:I

    iput v8, v1, Landroid/support/v4/app/BackStackRecord;->mBreadCrumbTitleRes:I

    .line 126
    iget-object v8, p0, Landroid/support/v4/app/BackStackState;->mBreadCrumbTitleText:Ljava/lang/CharSequence;

    iput-object v8, v1, Landroid/support/v4/app/BackStackRecord;->mBreadCrumbTitleText:Ljava/lang/CharSequence;

    .line 127
    iget v8, p0, Landroid/support/v4/app/BackStackState;->mBreadCrumbShortTitleRes:I

    iput v8, v1, Landroid/support/v4/app/BackStackRecord;->mBreadCrumbShortTitleRes:I

    .line 128
    iget-object v8, p0, Landroid/support/v4/app/BackStackState;->mBreadCrumbShortTitleText:Ljava/lang/CharSequence;

    iput-object v8, v1, Landroid/support/v4/app/BackStackRecord;->mBreadCrumbShortTitleText:Ljava/lang/CharSequence;

    .line 129
    invoke-virtual {v1, v11}, Landroid/support/v4/app/BackStackRecord;->bumpBackStackNesting(I)V

    .line 130
    return-object v1
.end method

.method public writeToParcel(Landroid/os/Parcel;I)V
    .registers 5
    .param p1, "dest"    # Landroid/os/Parcel;
    .param p2, "flags"    # I

    .prologue
    const/4 v1, 0x0

    .line 138
    iget-object v0, p0, Landroid/support/v4/app/BackStackState;->mOps:[I

    invoke-virtual {p1, v0}, Landroid/os/Parcel;->writeIntArray([I)V

    .line 139
    iget v0, p0, Landroid/support/v4/app/BackStackState;->mTransition:I

    invoke-virtual {p1, v0}, Landroid/os/Parcel;->writeInt(I)V

    .line 140
    iget v0, p0, Landroid/support/v4/app/BackStackState;->mTransitionStyle:I

    invoke-virtual {p1, v0}, Landroid/os/Parcel;->writeInt(I)V

    .line 141
    iget-object v0, p0, Landroid/support/v4/app/BackStackState;->mName:Ljava/lang/String;

    invoke-virtual {p1, v0}, Landroid/os/Parcel;->writeString(Ljava/lang/String;)V

    .line 142
    iget v0, p0, Landroid/support/v4/app/BackStackState;->mIndex:I

    invoke-virtual {p1, v0}, Landroid/os/Parcel;->writeInt(I)V

    .line 143
    iget v0, p0, Landroid/support/v4/app/BackStackState;->mBreadCrumbTitleRes:I

    invoke-virtual {p1, v0}, Landroid/os/Parcel;->writeInt(I)V

    .line 144
    iget-object v0, p0, Landroid/support/v4/app/BackStackState;->mBreadCrumbTitleText:Ljava/lang/CharSequence;

    invoke-static {v0, p1, v1}, Landroid/text/TextUtils;->writeToParcel(Ljava/lang/CharSequence;Landroid/os/Parcel;I)V

    .line 145
    iget v0, p0, Landroid/support/v4/app/BackStackState;->mBreadCrumbShortTitleRes:I

    invoke-virtual {p1, v0}, Landroid/os/Parcel;->writeInt(I)V

    .line 146
    iget-object v0, p0, Landroid/support/v4/app/BackStackState;->mBreadCrumbShortTitleText:Ljava/lang/CharSequence;

    invoke-static {v0, p1, v1}, Landroid/text/TextUtils;->writeToParcel(Ljava/lang/CharSequence;Landroid/os/Parcel;I)V

    .line 147
    return-void
.end method
