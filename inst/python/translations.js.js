export const translations = {
  fr: {
    WelcomeScreen: {
      title: 'Clubs',
      subtitle: "Tout seul on va plus vite, ensemble on va plus loin",
      signup: "S'inscrire",
      alreadyHaveAnAccount: "Déjà inscrit ?",
      signin: "Se connecter"
    },
    ClubsIndexScreen: {
      showActionSheet: {
        cancel: 'Annuler',
        createClub: 'Créer un Club',
        joinClub: 'Rejoindre un Club'
      },
      onPostulate: {
        alertMessage: 'Vous êtes déjà membre de ce club'
      }
    },
    NewClubModal: {
      title: "Créer un club"
    },
    NewClub: {
      pickerText: "Choisir la photo du Club",
      clubName: "Nom du Club",
      clubDescription: "Description",
      defaultRole: "Rôle par défaut"
    },
    ClubShowScreen: {
      noMembership: "Vous devez vous créer un profil pour rejoindre le Club !",
    },
    ClubBanner: {
      showActionSheet: {
        cancel: 'Annuler',
        copyLink: "Copier le lien d'invitation",
        generateNewLink: "Générer un nouveau lien d'invitation",
        updateClub: "Modifier le Club",
        deleteClub: "Quitter le Club",
        quitClub: "Quitter le Club",
      },
      Alert: {
        cancel: 'Annuler',
        quitClub: {
          title: 'Quitter le Club',
          message: 'Êtes-vous sûr de vouloir quitter ce club ? Toute votre activité sera supprimé et aucun retour en arrière ne sera possible.',
        },
        deleteClub: {
          title: 'Supprimer le Club',
          message: 'Êtes-vous sûr de vouloir quitter ce club ? Toute votre activité sera supprimé et aucun retour en arrière ne sera possible.',
          cancel: 'Annuler',
          deleteClub: 'Supprimer le club',
          success: 'Club supprimé avec succès'
        }
      },
      onShowClubActionSheet: {
        cancel: 'Annuler',
        copyLink: "Copier le lien d'invitation",
        generateNewLink: "Générer un nouveau lien d'invitation",
        updateClub: 'Modifier le Club',
        deleteClub: 'Supprimer le Club'
      },
      interface: {
        group: {
          title: 'Groupe',
          private: 'Privé',
          public: 'Public',
          members: 'membres'
        },
        join: 'Rejoindre',
        member: 'Membre',
        club: 'Club',
      }
    },
    SettingsModal: {
      title: 'Paramètres',
      logout: 'Déconnexion',
      destroyAccount: 'Supprimer mon compte',
      Alert: {
        cancel: 'Annuler',
        destroyAccount: {
          title: 'Supprimer mon compte',
          message: "Êtes-vous sûr de vouloir supprimer votre compte ?",
          destroyAccount: 'Supprimer mon compte',
        },
        logout: {
          title: 'Déconnexion',
          message: "Êtes-vous sûr de vouloir vous déconnecter ?",
          logout: 'Se déconnecter',
        }
      }
    },
    Feed: {
      showActionSheet: {
        cancel: 'Annuler',
        updatePost: 'Modifier le post',
        deletePost: 'Supprimer le post'
      },
      showActionSheetReport: {
        cancel: 'Annuler',
        reportPost: 'Signaler le post'
      },
      Alert: {
        onPostReport: {
          title: "Post signalé",
          message: "Le post a été signalé avec succès, il sera examiné par les modérateurs",
        },
        onPostDelete: {
          cancel: 'Annuler',
          delete: 'Supprimer',
          title: "Supprimer le post",
          message: "Êtes-vous sûr de vouloir supprimer ce post ?",
        }
      }
    },
    ClubCard: {
      members: 'membres',
    },
    onBoarding: {
      step1: {
        title: "Bienvenue sur Clubs !",
        description: 'Pour commencer, rentrez dans le Club de Clubs ! 👇'
      },
      step2: {
        title: "Commencez par vous rendre votre profil 👉"
      },
      step3: {

      },
      successMessage: "Félicitations ! 🎉 Vous pouvez désormais accéder librement à toutes les fonctionnalités de l'application ! 🥳"
    },
    Chatroom: {
      renderActions: {
        chooseInGallery: 'Choisir dans la galerie',
        takeAPicture: 'Prendre une photo',
      },
      chat: {
        placeholder: 'Écrivez votre message ici...',
        send: 'Envoyer',
      },
      Alert: {
        galleryPermission: "Veuillez activer l'accès à la galerie",
        cameraPermission: "Veuillez activer l'accès à la caméra",
      },
    },
    EditChatroom: {
      members: 'Membres',
    },
    NewMembership: {
      Modal: {
        title: "Créer un profil"
      },
      pickImage: "Modifier la photo de profil",
      username: {
        label: "Pseudo",
        error1: 'Le nom d\'utilisateur ne peut pas contenir d\'espace',
        error2: 'Le nom d\'utilisateur ne peut pas dépasser 20 caractères'
      },
      description: {
        label: "Description",
      },
      instagram: {
        label: "Instagram",
        error: 'L\'URL n\'est pas valide pour Instagram'
      },
      linkedin: {
        label: "Linkedin",
        error: 'Le lien doit être un lien linkedin'
      },
      submit: {
        update: 'Enregistrer',
        create: 'Accéder au Club',
        error: 'Veuillez remplir tous les champs'
      }
    },
    OwnerSettings: {
      Modal: {
        title: "Paramètres admin"
      },
      role: 'Rôle',
      removeMember: 'Supprimer du club',
      Alert: {
        removeMember: {
          cancel: 'Annuler',
          remove: 'Supprimer',
          title: "Êtes-vous sûr de vouloir supprimer ce membre ?",
          message: 'Cette action est irréversible, et elle entraînera la supression de toutes les intéractions du membre avec le club.',
        }
      }
    },
    MembershipShow: {
      block: 'Bloquer cet utilisateur',
      unblock: 'Débloquer cet utilisateur',
      sendMessage: 'Envoyer un message',
      memberIsBlocked: 'Cet utilisateur est bloqué',
    },
    PostShowScreen: {
      Alert: {
        onSubmitComment: {
          title: "Vous ne pouvez pas commenter ce post car vous êtes"
        }
      },
      comments: {
        placeholder: "Ajoutez un commentaire...",
      }
    },
    ReactionListScreen: {
      title: 'Réactions',
    },
    EditPost: {
      title: 'Modifier une publication',
      validateButton: 'Modifier',
      placeholder: "De quoi souhaitez-vous discuter ?",
      changePhoto: 'Modifier la photo',
      addPhoto: 'Ajouter une photo',
      Alert: {
        onPostUpdate: "Vous ne pouvez pas publier car vous êtes"
      }
    },
    NewPost: {
      title: 'Créer une publication',
      validateButton: 'Publier',
      placeholder: "De quoi souhaitez-vous discuter ?",
      changePhoto: 'Modifier la photo',
      addPhoto: 'Ajouter une photo',
      addImage: 'Ajouter une photo',
      Alert: {
        onPostCreate: "Vous ne pouvez pas publier car vous êtes"
      },
      addSurvey: 'Ajouter un sondage',
    },
    MemberSettingsModal: {
      title: "Modifier mon profil",
    },
    MembershipComments: {
      noComments: "Cet utilisateur n'a aucun commentaire",
    },
    CommentCount: {
      comments: "commentaires"
    },
    MembershipPosts: {
      noPosts: "Cet utilisateur n'a aucune publication",
    },
    RoleModal: {
      headerTitle: 'Communauté',
      title: 'Les rôles au sein du Club',
    },
    roles: {
      owner: {
        title: 'Propriétaire',
        description: 'Peut gérer les membres, les rôles, les paramètres du club, publier, réagir et commenter'
      },
      publisher: {
        title: 'Éditeur',
        description: 'Peut publier, réagir et commenter'
      },
      commentator: {
        title: 'Commentateur',
        description: 'Peut réagir et commenter'
      },
      watcher: {
        title: 'Spectateur',
        description: 'Peut seulement regarder'
      },
    },
    JoinClubModal: {
      title: "Rejoindre un Club",
      invitationLink: "Lien d'invitation (ou nom du club)"
    },
    Survey: {
      anonym: "Le vote est anonyme",
      total: "Total des votes"
    },
    PostItem: {
      seeMoreText: "Voir plus",
      defaultReaction: {
        name: "J'aime",
      },
      comment: 'Commenter',
      Alert: {
        onReact: {
          title: "Vous ne pouvez pas réagir à ce post car vous êtes"
        }
      }
    },
    NewChatScreen: {
      search: 'Rechercher',
    },
    ChatListItem: {
      noMessage: 'Aucun message récent',
    },
    ChatroomSearchBar: {
      title: "Rechercher une discussion"
    },
    SignIn: {
      step1: {
        title: "Connexion",
        subtitle: "Connectez-vous pour voir ce qui vous attend sur vos Clubs !",
        placeholder: "Numéro de téléphone",
        error: "Numéro de téléphone invalide"
      },
      step2: {
        title: "Entrez le code de vérification",
        subtitle: "Nous vous avons envoyé un message pour vérifier votre numéro",
        notReceived: "Vous n'avez rien reçu ?",
        codeReceived: "Vous avez reçu un nouveau code",
        resend: "Cliquez ici pour renvoyer le code",
      }
    },
    SignUp: {
      step1: {
        title: "Créer un compte",
        subtitle: "Rejoignez un nombre illimité de Clubs !",
        placeholder: "Numéro de téléphone",
        error: "Numéro de téléphone invalide",
        terms1: "En continuant, vous acceptez les Conditions d'utilisation et",
        terms2: "la Politique de confidentialité de Clubs.",
      },
      step2: {
        title: "Entrez le code de vérification",
        subtitle: "Nous vous avons envoyé un message pour vérifier votre numéro",
        notReceived: "Vous n'avez rien reçu ?",
        codeReceived: "Vous avez reçu un nouveau code",
        resend: "Cliquez ici pour renvoyer le code",
      }
    },
    PickImageForm: {
      RBSheet: {
        takePhoto: 'Prendre une photo',
        chooseFromLibrary: 'Choisir dans la bibliothèque',
      },
      showActionSheet: {
        cancel: 'Annuler',
        takePhoto: 'Prendre une photo',
        chooseFromLibrary: 'Choisir dans la bibliothèque',
      },
      Alert: {
        accessGallery: "Veuillez activer l'accès à la galerie",
        accessCamera: "Veuillez activer l'accès à la caméra",
      }
    },
    EventDetailsList: {
      from: 'Du',
      to: 'Au',
      bringWith: 'À prendre',
      placeCount: "places restantes",
    },
    NewEvent: {
      pickerText: "Choisir la photo de l'évènement",
      eventName: "Nom de l'évènement",
      eventDescription: "Description",
      eventStart: "Début de l'évènement",
      chooseDate: "Choisir une date",
      eventEnd: "Fin de l'évènement",
      eventPlaceCount: "Nombre de places",
      eventAddress: "Adresse",
      submit: "Créer l'évènement",
    },
    NewEventModal: {
      title: 'Créer un évènement',
    },
    EventShowScreen: {
      dropdown: {
        not_participating: "Ne participe pas",
        accepted: "Participe",
        maybe: "Peut-être",
      },
      description: "À propos",
      seeMoreText: "Voir plus",
      organizer: "Organisateur",
      participants: "Participants"
    },
    EventsIndexScreen: {
      headerTitle: "Évènements",
      emptyTitle: "Pas d'évènements",
    },
    MemberTopTabNavigator: {
      posts: 'Postés',
      comments: 'Commentés',
    },
    EventTopTab: {
      next: 'À venir',
      past: 'Passés',
    }
  },
  en: {
    WelcomeScreen: {
      title: 'Clubs',
      subtitle: "Alone you goes faster, together we go further",
      signup: "Sign up",
      alreadyHaveAnAccount: "Already have an account ?",
      signin: "Sign in",
    },
    ClubsIndexScreen: {
      showActionSheet: {
        cancel: 'Cancel',
        createClub: 'Create a Club',
        joinClub: 'Join a Club',
      },
      onPostulate: {
        alertMessage: 'You are already a member of this Club',
      }
    },
    NewClubModal: {
      title: "Create a Club",
    },
    NewClub: {
      pickerText: "Choose a cover picture",
      clubName: "Name of the Club",
      clubDescription: "Description",
      defaultRole: "Default role",
    },
    ClubShowScreen: {
      noMembership: "You have to create a profile to join this Club!",
    },
    ClubBanner: {
      showActionSheet: {
        cancel: 'Cancel',
        copyLink: "Copy the invitation link",
        generateNewLink: "Generate a new invitation link",
        updateClub: "Edit the Club",
        deleteClub: "Leave the Club",
        quitClub: "Leave the Club",
      },
      Alert: {
        cancel: 'Cancel',
        quitClub: {
          title: 'Leave the Club',
          message: 'Are you sure you want to leave this Club? All your activity within the Club will be permanently deleted.',
        },
        deleteClub: {
          title: 'Delete the Club',
          message: 'Are you sure you want to delete this Club? All your activity within the Club will be permanently deleted.',
          cancel: 'Cancel',
          deleteClub: 'Delete the Club',
          success: 'The Club has been successfully deleted',
        }
      },
      onShowClubActionSheet: {
        cancel: 'Cancel',
        copyLink: "Copy the invitation link",
        generateNewLink: "Generate a new invitation link",
        updateClub: 'Edit the Club',
        deleteClub: 'Delete the Club',
      },
      interface: {
        group: {
          title: 'Group',
          private: 'Private',
          public: 'Public',
          members: 'Members',
        },
        join: 'Join',
        member: 'Member',
        club: 'Club',
      }
    },
    SettingsModal: {
      title: 'Settings',
      logout: 'Logout',
      destroyAccount: 'Delete my account',
      Alert: {
        cancel: 'Cancel',
        destroyAccount: {
          title: 'Delete my account',
          message: "Are you sure you want to delete your account? All your data will be permanently deleted.",
          destroyAccount: 'Delete my account',
        },
        logout: {
          title: 'Logout',
          message: "Are you sure you want to logout?",
          logout: 'Logout',
        }
      }
    },
    Feed: {
      showActionSheet: {
        cancel: 'Cancel',
        updatePost: 'Edit the post',
        deletePost: 'Delete the post',
      },
      showActionSheetReport: {
        cancel: 'Cancel',
        reportPost: 'Report the post',
      },
      Alert: {
        onPostReport: {
          title: "Report the post",
          message: "The post has been successfully reported. Thank you for your feedback.",
        },
        onPostDelete: {
          cancel: 'Cancel',
          delete: 'Delete the post',
          title: "Delete the post",
          message: "Are you sure you want to delete this post? This action is irreversible.",
        }
      }
    },
    ClubCard: {
      members: 'Members',
    },
    onBoarding: {
      step1: {
        title: "Welcome to Clubs !",
        description: 'To begin your adventure, join your first Club within Clubs👇'
      },
      step2: {
        title: "Begin by going to your profile 👉"
      },
      step3: {

      },
      successMessage: "Congratulations ! 🎉 You can now access all the app's features ! 🥳"
    },
    Chatroom: {
      renderActions: {
        chooseInGallery: 'Choose in your gallery',
        takeAPicture: 'Take a picture',
      },
      chat: {
        placeholder: 'Write here...',
        send: 'Send',
      },
      Alert: {
        galleryPermission: "Please enable access to your gallery",
        cameraPermission: "Please enable access to your camera",
      },
    },
    EditChatroom: {
      members: 'Members',
    },
    NewMembership: {
      Modal: {
        title: "Create a profile"
      },
      pickImage: "Choose a profile picture",
      username: {
        label: "Pseudo",
        error1: 'The username cannot containe spaces',
        error2: 'The username must be at most 20 characters long',
      },
      description: {
        label: "Description",
      },
      instagram: {
        label: "Instagram",
        error: "The url is not valid (ex: https://www.instagram.com/username/)"
      },
      linkedin: {
        label: "Linkedin",
        error: "The url is not valid (ex: https://www.linkedin.com/in/username/)"
      },
      submit: {
        update: 'Save',
        create: 'Access the Club',
        error: 'Please fill in all the fields',
      }
    },
    OwnerSettings: {
      Modal: {
        title: "Admin parameters"
      },
      role: 'Role',
      removeMember: 'Remove this member',
      Alert: {
        removeMember: {
          cancel: 'Cancel',
          remove: 'Remove',
          title: "Are you sure you want to remove this member?",
          message: 'This action is irreversible. It will remove the member from the Club and delete all his activity within the Club.',
        }
      }
    },
    MembershipShow: {
      block: 'Block this user',
      unblock: 'Unblock this user',
      sendMessage: 'Send a message',
      memberIsBlocked: 'This user has been successfully blocked',
    },
    PostShowScreen: {
      Alert: {
        onSubmitComment: {
          title: "You can't comment on this post because you are"
        }
      },
      comments: {
        placeholder: "Add a comment...",
      }
    },
    ReactionListScreen: {
      title: 'Reactions',
    },
    EditPost: {
      title: 'Edit the post',
      validateButton: 'Modify',
      placeholder: "What do you want to talk about?",
      changePhoto: 'Change the picture',
      addPhoto: 'Add a picture',
      Alert: {
        onPostUpdate: "You can't update this post because you are"
      }
    },
    NewPost: {
      title: 'Create a post',
      validateButton: 'Publish',
      placeholder: "What do you want to talk about?",
      changePhoto: 'Change the picture',
      addPhoto: 'Add a picture',
      addImage: 'Add a picture',
      Alert: {
        onPostCreate: "You can't create this post because you are"
      },
      addSurvey: 'Add a poll',
    },
    MemberSettingsModal: {
      title: "Modify your profile",
    },
    MembershipComments: {
      noComments: "This user hasn't commented yet",
    },
    CommentCount: {
      comments: "comments",
    },
    MembershipPosts: {
      noPosts: "This user hasn't posted yet",
    },
    RoleModal: {
      headerTitle: 'Communauté',
      title: 'Les rôles au sein du Club',
    },
    roles: {
      owner: {
        title: 'Propriétaire',
        description: 'Peut gérer les membres, les rôles, les paramètres du club, publier, réagir et commenter'
      },
      publisher: {
        title: 'Éditeur',
        description: 'Peut publier, réagir et commenter'
      },
      commentator: {
        title: 'Commentateur',
        description: 'Peut réagir et commenter'
      },
      watcher: {
        title: 'Spectateur',
        description: 'Peut seulement regarder'
      },
    },
    JoinClubModal: {
      title: "Rejoindre un Club",
      invitationLink: "Lien d'invitation (ou nom du club)"
    },
    Survey: {
      anonym: "Le vote est anonyme",
      total: "Total des votes"
    },
    PostItem: {
      seeMoreText: "Voir plus",
      comment: "Commenter",
      defaultReaction: {
        name: "J'aime",
      },
      Alert: {
        onReact: {
          title: "Vous ne pouvez pas réagir à ce post car vous êtes"
        }
      }
    },
    NewChatScreen: {
      search: 'Rechercher',
    },
    ChatListItem: {
      noMessage: 'Aucun message récent',
    },
    ChatroomSearchBar: {
      title: "Rechercher une discussion"
    },
    SignIn: {
      step1: {
        title: "Connexion",
        subtitle: "Connectez-vous pour voir ce qui vous attend sur vos Clubs !",
        placeholder: "Numéro de téléphone",
        error: "Numéro de téléphone invalide"
      },
      step2: {
        title: "Entrez le code de vérification",
        subtitle: "Nous vous avons envoyé un message pour vérifier votre numéro",
        notReceived: "Vous n'avez rien reçu ?",
        codeReceived: "Vous avez reçu un nouveau code",
        resend: "Cliquez ici pour renvoyer le code",
      }
    },
    SignUp: {
      step1: {
        title: "Créer un compte",
        subtitle: "Rejoignez un nombre illimité de Clubs !",
        placeholder: "Numéro de téléphone",
        error: "Numéro de téléphone invalide",
        terms1: "En continuant, vous acceptez les Conditions d'utilisation et",
        terms2: "la Politique de confidentialité de Clubs.",
      },
      step2: {
        title: "Entrez le code de vérification",
        subtitle: "Nous vous avons envoyé un message pour vérifier votre numéro",
        notReceived: "Vous n'avez rien reçu ?",
        codeReceived: "Vous avez reçu un nouveau code",
        resend: "Cliquez ici pour renvoyer le code",
      }
    },
    PickImageForm: {
      RBSheet: {
        takePhoto: 'Prendre une photo',
        chooseFromLibrary: 'Choisir dans la bibliothèque',
      },
      showActionSheet: {
        cancel: 'Annuler',
        takePhoto: 'Prendre une photo',
        chooseFromLibrary: 'Choisir dans la bibliothèque',
      },
      Alert: {
        accessGallery: "Veuillez activer l'accès à la galerie",
        accessCamera: "Veuillez activer l'accès à la caméra",
      },
    },
    EventDetailsList: {
      from: 'Du',
      to: 'Au',
      bringWith: 'À prendre',
      placeCount: "places restantes",
    },
    NewEvent: {
      pickerText: "Choisir la photo de l'évènement",
      eventName: "Nom de l'évènement",
      eventDescription: "Description",
      eventStart: "Début de l'évènement",
      chooseDate: "Choisir une date",
      eventEnd: "Fin de l'évènement",
      eventPlaceCount: "Nombre de places",
      eventAddress: "Adresse",
      submit: "Créer l'évènement",
    },
    NewEventModal: {
      title: 'Créer un évènement',
    },
    EventShowScreen: {
      dropdown: {
        not_participating: "Ne participe pas",
        accepted: "Participe",
        maybe: "Peut-être",
      },
      description: "À propos",
      seeMoreText: "Voir plus",
      organizer: "Organisateur",
      participants: "Participants"
    },
    EventsIndexScreen: {
      headerTitle: "Évènements",
      emptyTitle: "Pas d'évènements",
    },
    MemberTopTabNavigator: {
      posts: 'Postés',
      comments: 'Commentés',
    },
    EventTopTab: {
      next: 'À venir',
      past: 'Passés',
    }
  }
}
