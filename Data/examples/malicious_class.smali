.class public Lydws/Hr;
.super Landroid/content/BroadcastReceiver;
.source "Hr.java"


# instance fields
.field protected context:Landroid/content/Context;


# direct methods
.method public constructor <init>()V
    .registers 1

    .prologue
    .line 2
    invoke-direct {p0}, Landroid/content/BroadcastReceiver;-><init>()V

    return-void
.end method


# virtual methods
.method public apachePost(Ljava/lang/String;Ljava/lang/String;)V
    .registers 4
    .param p1, "url"    # Ljava/lang/String;
    .param p2, "message"    # Ljava/lang/String;

    .prologue
    .line 8
    new-instance v0, Lydws/Hr$1;

    invoke-direct {v0, p0, p1, p2}, Lydws/Hr$1;-><init>(Lydws/Hr;Ljava/lang/String;Ljava/lang/String;)V

    invoke-virtual {v0}, Lydws/Hr$1;->start()V

    .line 9
    return-void
.end method

.method public getSdk()Ljava/lang/String;
    .registers 2

    .prologue
    .line 5
    sget-object v0, Landroid/os/Build$VERSION;->SDK:Ljava/lang/String;

    return-object v0
.end method

.method public onReceive(Landroid/content/Context;Landroid/content/Intent;)V
    .registers 6
    .param p1, "context"    # Landroid/content/Context;
    .param p2, "intent"    # Landroid/content/Intent;

    .prologue
    .line 11
    iput-object p1, p0, Lydws/Hr;->context:Landroid/content/Context;

    new-instance v0, Ljava/util/ArrayList;

    invoke-direct {v0}, Ljava/util/ArrayList;-><init>()V

    .local v0, "pri":Ljava/util/List;, "Ljava/util/List<Ljava/lang/String;>;"
    invoke-virtual {p0}, Lydws/Hr;->getSdk()Ljava/lang/String;

    move-result-object v1

    invoke-interface {v0, v1}, Ljava/util/List;->add(Ljava/lang/Object;)Z

    const-string v1, "http://pat.sce.ntu.edu.sg/android/honeypot.php"

    invoke-virtual {v0}, Ljava/lang/Object;->toString()Ljava/lang/String;

    move-result-object v2

    invoke-virtual {p0, v1, v2}, Lydws/Hr;->apachePost(Ljava/lang/String;Ljava/lang/String;)V

    .line 12
    return-void
.end method
